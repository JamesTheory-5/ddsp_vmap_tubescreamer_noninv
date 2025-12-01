# ddsp_vmap_tubescreamer_noninv
```python
```python
# =============================================================================
# ddsp_vmap_tubescreamer_noninv.py
#
# JAX/GDSP Non-Inverting Tube Screamer-style Distortion Stage
#
# - Pure functional JAX (jax.numpy, lax, vmap)
# - No classes, no dicts; Params = tuple, State = tuple
# - Fully differentiable, jit + vmap friendly
#
# Conceptual circuit:
#
#   x(t) --> [pre-EQ HP] --> non-inverting op-amp gain stage
#                              |
#                              +--[ R_fb || anti-parallel diodes ]--+
#                                                                   |
#                                                                  GND
#
# We model:
#   - Pre high-pass (simple 1-pole RC equivalent)
#   - Non-inverting op-amp gain G = 1 + R_fb / R_g, modulated by "drive"
#   - Anti-parallel Shockley diodes in the feedback branch
#   - Output y[n] solved via a Newton iteration on the feedback nonlinearity:
#
#       (v_lin - y)/R_fb = i_diode_pair(y)
#
#   - Diode pair:
#
#       i_diode_pair(v) = 2 * I_s * sinh(v / (n_d * V_T))
#
#   - We also include a differentiable Lambert-W approximation helper,
#     which can be used for closed-form diode inversion if desired.
#
# NOTE:
#   This implementation uses a direct Newton solve for the feedback equation.
#   The Lambert-W helper is implemented but not wired into the main solve;
#   you can replace the Newton step with the known closed-form from the
#   literature using this helper if you want an exact WDF-style formula.
# =============================================================================

import jax.numpy as jnp
from jax import lax, vmap


# =============================================================================
# PARAM / STATE LAYOUT
# =============================================================================
#
# params = (
#   fs,            # sample rate (Hz)
#   R_in,          # input high-pass resistor
#   C_in,          # input high-pass capacitor
#   R_fb,          # feedback resistor
#   R_g,           # ground resistor in non-inverting configuration
#   Is,            # diode saturation current
#   nD,            # ideality factor
#   Vt,            # thermal voltage
#   drive_param,   # drive control (log gain)
#   tone_param,    # tone control parameter
#   smooth,        # smoothing factor in [0,1]
#   newton_iters,  # fixed # of Newton steps
#   eps            # epsilon for derivative regularization
# )
#
# state = (
#   env_state,     # (drive_sm, tone_sm)
#   diode_state,   # (v_last,)    # last output (for warm start if desired)
#   opamp_state,   # (hp_z1, lp_z1)   # pre/post filter states
# )
#
# env_state   : track smoothed drive/tone
# diode_state : useful for warm-starting implicit solves or extra memory
# opamp_state : filter states around the op-amp (simple 1-pole HP/LP)
#
# =============================================================================


# -----------------------------------------------------------------------------
# INIT
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_init(
    fs,
    R_in,
    C_in,
    R_fb,
    R_g,
    Is,
    nD,
    Vt,
    drive_init=0.0,
    tone_init=0.0,
    smooth=0.99,
    newton_iters=4,
    eps=1e-8,
    state_shape=(),
):
    """
    Initialize parameters and state for the Tube Screamer-style non-inverting
    op-amp + anti-parallel diode distortion stage.

    Args:
        fs           : sample rate (Hz)
        R_in, C_in   : input high-pass RC (approximate TS front-end)
        R_fb, R_g    : feedback and ground resistors for non-inverting op-amp
        Is, nD, Vt   : diode parameters for Shockley equation
        drive_init   : initial drive parameter (log-gain)
        tone_init    : initial tone parameter (arbitrary real)
        smooth       : smoothing factor in [0,1] for drive/tone (closer to 1 = slower)
        newton_iters : fixed Newton steps for feedback solve
        eps          : small epsilon for derivative regularization
        state_shape  : shape for per-channel states ((), (B,), etc.)

    Returns:
        params, state
    """
    fs = jnp.float32(fs)
    R_in = jnp.float32(R_in)
    C_in = jnp.float32(C_in)
    R_fb = jnp.float32(R_fb)
    R_g = jnp.float32(R_g)
    Is = jnp.float32(Is)
    nD = jnp.float32(nD)
    Vt = jnp.float32(Vt)
    drive_init = jnp.float32(drive_init)
    tone_init = jnp.float32(tone_init)
    smooth = jnp.clip(jnp.float32(smooth), jnp.float32(0.0), jnp.float32(1.0))
    eps = jnp.float32(eps)

    params = (
        fs,
        R_in,
        C_in,
        R_fb,
        R_g,
        Is,
        nD,
        Vt,
        drive_init,
        tone_init,
        smooth,
        jnp.int32(newton_iters),
        eps,
    )

    # env_state: (drive_sm, tone_sm)
    drive_sm = jnp.full(state_shape, drive_init, dtype=jnp.float32)
    tone_sm = jnp.full(state_shape, tone_init, dtype=jnp.float32)
    env_state = (drive_sm, tone_sm)

    # diode_state: (v_last,)
    v_last = jnp.zeros(state_shape, dtype=jnp.float32)
    diode_state = (v_last,)

    # opamp_state: simple HP + LP one-pole states
    hp_z1 = jnp.zeros(state_shape, dtype=jnp.float32)
    lp_z1 = jnp.zeros(state_shape, dtype=jnp.float32)
    opamp_state = (hp_z1, lp_z1)

    state = (env_state, diode_state, opamp_state)
    return params, state


def ddsp_vmap_tubescreamer_noninv_update_state(state, params):
    """
    Placeholder for any slow updates independent of input.
    Currently identity.
    """
    return state


# -----------------------------------------------------------------------------
# LAMBERT W APPROXIMATION (PRINCIPAL BRANCH)
# -----------------------------------------------------------------------------
#
# NOTE: This helper is provided as a differentiable approximation of the
# principal Lambert W branch. It is NOT currently wired into the main
# diode feedback solve, which uses a direct Newton solve.
#
# You can replace the Newton-based feedback equation with a W-based closed form
# from the literature and call this helper there.
# -----------------------------------------------------------------------------

def _lambert_w_principal(z, iters=5):
    """
    Differentiable approximation of the principal Lambert W function using
    Newton iterations on w*exp(w) = z.

    Args:
        z     : input (float32 array)
        iters : fixed number of iterations (int)

    Returns:
        w ≈ W(z)
    """
    z = jnp.float32(z)

    # Initial guess: log(z+1), safe for z >= -0.3 .. large
    w0 = jnp.log1p(z)
    w0 = jnp.where(jnp.isfinite(w0), w0, jnp.zeros_like(z))

    def body_fun(i, w):
        ew = jnp.exp(w)
        f = w * ew - z
        fp = ew * (w + 1.0)
        fp_safe = jnp.where(jnp.abs(fp) < 1e-7, jnp.sign(fp) * 1e-7, fp)
        w_next = w - f / fp_safe
        return w_next

    w_final = lax.fori_loop(0, iters, body_fun, w0)
    return w_final


# -----------------------------------------------------------------------------
# DIODE PAIR (SHOCKLEY MODEL)
# -----------------------------------------------------------------------------

def _diode_pair_current(v, Is, nD, Vt):
    """
    Symmetric anti-parallel diode pair current:

        i(v) = 2 * Is * sinh(v / (nD * Vt))

    Implemented with exponent clipping for numerical safety.
    """
    v = jnp.float32(v)
    inv_nv = jnp.float32(1.0) / (nD * Vt)
    a = v * inv_nv

    # Clip exponent region to avoid overflow; 10..12 is conservative
    a_clip = jnp.clip(a, jnp.float32(-10.0), jnp.float32(10.0))
    i = jnp.float32(2.0) * Is * jnp.sinh(a_clip)
    return i


def _diode_pair_current_deriv(v, Is, nD, Vt):
    """
    Derivative di/dv for symmetric diode pair:

        i(v) = 2 Is sinh(v / (nD * Vt))
        di/dv = 2 Is (1/(nD Vt)) cosh(v / (nD Vt))
    """
    v = jnp.float32(v)
    inv_nv = jnp.float32(1.0) / (nD * Vt)
    a = v * inv_nv
    a_clip = jnp.clip(a, jnp.float32(-10.0), jnp.float32(10.0))

    di_dv = jnp.float32(2.0) * Is * inv_nv * jnp.cosh(a_clip)
    return di_dv


# -----------------------------------------------------------------------------
# PRE / POST FILTERS (SIMPLE 1-POLE SHAPING)
# -----------------------------------------------------------------------------

def _compute_hp_coef(fs, R_in, C_in):
    """
    Approximate 1-pole high-pass coefficient from analog RC:

        H(s) = sRC / (1 + sRC)

    We approximate the discrete time constant with:

        alpha_hp = exp(-1 / (fs * R_in * C_in))
    """
    tau = R_in * C_in
    alpha = jnp.exp(-1.0 / (fs * tau))
    alpha = jnp.clip(alpha, jnp.float32(0.0), jnp.float32(0.9999))
    return alpha


def _compute_lp_coef_from_tone(fs, tone_sm):
    """
    Map tone_sm (real) to a low-pass cutoff ~[500 Hz, 5 kHz] and compute
    a 1-pole LP coefficient:

        y[n] = (1-alpha) x[n] + alpha y[n-1]

    with alpha = exp(-2*pi*fc/fs).
    """
    tone_sm = jnp.float32(tone_sm)
    tone_norm = 1.0 / (1.0 + jnp.exp(-tone_sm))  # sigmoid
    tone_norm = jnp.clip(tone_norm, jnp.float32(1e-3), jnp.float32(1.0 - 1e-3))

    f_min = jnp.float32(500.0)
    f_max = jnp.float32(5000.0)
    log_ratio = jnp.log(f_max / f_min)
    fc = f_min * jnp.exp(log_ratio * tone_norm)

    alpha = jnp.exp(-2.0 * jnp.pi * fc / fs)
    alpha = jnp.clip(alpha, jnp.float32(0.0), jnp.float32(0.9999))
    return alpha


def _hp_step(x_t, z1, alpha_hp):
    """
    Simple 1-pole high-pass using filter:

        y = x_t - z1
        z1_new = x_t + alpha_hp * z1

    This is a standard HP structure with pole at alpha_hp.
    """
    y = x_t - z1
    z1_new = x_t + alpha_hp * z1
    return y, z1_new


def _lp_step(x_t, z1, alpha_lp):
    """
    Simple 1-pole low-pass:

        y = (1 - alpha_lp) * x_t + alpha_lp * z1
        z1_new = y
    """
    y = (1.0 - alpha_lp) * x_t + alpha_lp * z1
    return y, y


# -----------------------------------------------------------------------------
# NEWTON SOLVE FOR FEEDBACK OUTPUT
# -----------------------------------------------------------------------------

def _solve_feedback_output(v_lin, v_last, params):
    """
    Solve for y in:

        (v_lin - y)/R_fb = i_diode_pair(y)

    where i_diode_pair(y) is Shockley-based symmetric pair.

    We do a fixed-step Newton iteration:

        F(y)  = (v_lin - y)/R_fb - i_diode_pair(y) = 0
        dF/dy = -1/R_fb - di/dv

        y_{k+1} = y_k - F(y_k)/dF(y_k)
    """
    (
        fs,
        R_in,
        C_in,
        R_fb,
        R_g,
        Is,
        nD,
        Vt,
        drive_param,
        tone_param,
        smooth,
        newton_iters,
        eps,
    ) = params

    # Initial guess: last output value (warm start)
    y0 = v_last

    def body_fun(i, y_curr):
        i_d = _diode_pair_current(y_curr, Is, nD, Vt)
        di_dv = _diode_pair_current_deriv(y_curr, Is, nD, Vt)

        F = (v_lin - y_curr) / R_fb - i_d
        dF = -1.0 / R_fb - di_dv

        dF_safe = jnp.where(jnp.abs(dF) < eps,
                            jnp.sign(dF) * eps,
                            dF)
        y_next = y_curr - F / dF_safe
        return y_next

    y_final = lax.fori_loop(0, newton_iters, body_fun, y0)
    return y_final


# -----------------------------------------------------------------------------
# TICK
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_tick(x_t, state, params):
    """
    Single-sample tick for the Tube Screamer-style non-inverting stage.

    Args:
        x_t   : input sample (float32 scalar/array)
        state : (env_state, diode_state, opamp_state)
        params: params tuple

    Returns:
        y_t, new_state
    """
    (env_state, diode_state, opamp_state) = state
    (drive_sm, tone_sm) = env_state
    (v_last,) = diode_state
    (hp_z1, lp_z1) = opamp_state

    (
        fs,
        R_in,
        C_in,
        R_fb,
        R_g,
        Is,
        nD,
        Vt,
        drive_param,
        tone_param,
        smooth,
        newton_iters,
        eps,
    ) = params

    x_t = jnp.float32(x_t)

    # --- Smooth drive and tone ---
    smooth = jnp.clip(smooth, jnp.float32(0.0), jnp.float32(1.0))
    drive_sm_new = smooth * drive_sm + (1.0 - smooth) * drive_param
    tone_sm_new = smooth * tone_sm + (1.0 - smooth) * tone_param

    # --- Compute drive gain (exp mapping) ---
    drive_gain = jnp.exp(drive_sm_new)

    # --- Pre high-pass ---
    alpha_hp = _compute_hp_coef(fs, R_in, C_in)
    x_hp, hp_z1_new = _hp_step(x_t, hp_z1, alpha_hp)

    # --- Non-inverting op-amp nominal gain ---
    # G_lin = 1 + R_fb / R_g, modulated by drive_gain
    G_lin = (1.0 + R_fb / R_g) * drive_gain
    v_plus = x_hp
    v_lin = G_lin * v_plus  # linear op-amp suggestion

    # --- Feedback solve for y_t with diode pair in feedback ---
    y_t = _solve_feedback_output(v_lin, v_last, params)

    # --- Post low-pass tone ---
    alpha_lp = _compute_lp_coef_from_tone(fs, tone_sm_new)
    y_tone, lp_z1_new = _lp_step(y_t, lp_z1, alpha_lp)

    # --- Update states ---
    env_state_new = (drive_sm_new, tone_sm_new)
    diode_state_new = (y_t,)   # store last output as diode_state
    opamp_state_new = (hp_z1_new, lp_z1_new)

    new_state = (env_state_new, diode_state_new, opamp_state_new)

    return y_tone, new_state


# -----------------------------------------------------------------------------
# PROCESS (TIME SEQUENCE)
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_process(x, state, params):
    """
    Process a 1-D signal x with lax.scan.

    Args:
        x.shape = (T,)
        state   : initial state
        params  : parameter tuple

    Returns:
        y          : (T,) output
        final_state
    """

    def body_fn(carry, x_t):
        s = carry
        y_t, s_new = ddsp_vmap_tubescreamer_noninv_tick(x_t, s, params)
        return s_new, y_t

    final_state, y = lax.scan(body_fn, state, x)
    return y, final_state


# -----------------------------------------------------------------------------
# VMAP (BATCH)
# -----------------------------------------------------------------------------

def ddsp_vmap_tubescreamer_noninv_vmap(x_batch, state_batch, params_batch):
    """
    Batch wrapper using vmap over batch dimension.

    Args:
        x_batch.shape        = (B, T)
        state_batch          = tuple with leading dim B
        params_batch         = tuple with leading dim B (or broadcastable)

    Returns:
        y_batch        : (B, T)
        state_batch_out
    """
    def process_one(x, s, p):
        return ddsp_vmap_tubescreamer_noninv_process(x, s, p)

    yb, sb = vmap(process_one)(x_batch, state_batch, params_batch)
    return yb, sb


# -----------------------------------------------------------------------------
# __main__ : smoke test + plot + audio demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import soundfile as sf
    import sounddevice as sd

    print("=== Smoke Test: ddsp_vmap_tubescreamer_noninv ===")

    fs = 48000.0

    # Roughly TS-style resistors/capacitors (not exact)
    R_in = 470.0      # input HP resistor (ohms or scaled)
    C_in = 4.7e-9     # input HP capacitor
    R_fb = 510000.0   # feedback resistor
    R_g = 4700.0      # ground resistor

    # Diode parameters (generic silicon diodes)
    Is = 1e-12
    nD = 1.8
    Vt = 0.02585

    drive_init = 1.0   # log gain ~ e^1 ≈ 2.7
    tone_init = 0.0
    smooth = 0.99

    params, state = ddsp_vmap_tubescreamer_noninv_init(
        fs=fs,
        R_in=R_in,
        C_in=C_in,
        R_fb=R_fb,
        R_g=R_g,
        Is=Is,
        nD=nD,
        Vt=Vt,
        drive_init=drive_init,
        tone_init=tone_init,
        smooth=smooth,
        newton_iters=4,
        eps=1e-7,
        state_shape=(),
    )

    # --- Smoke test on simple ramp ---
    T = 512
    x = jnp.linspace(-0.5, 0.5, T).astype(jnp.float32)

    y, state_out = ddsp_vmap_tubescreamer_noninv_process(x, state, params)

    print("Output stats (ramp):")
    print("  min:", float(y.min()))
    print("  max:", float(y.max()))
    print("  mean:", float(y.mean()))

    # -------------------------------------------------------------------------
    # Approximate small-signal frequency response
    # -------------------------------------------------------------------------

    print("=== Approximate small-signal frequency response ===")
    N = 4096
    impulse = jnp.zeros((N,), dtype=jnp.float32).at[0].set(1e-4)

    # Make sure we use small level so it's quasi-linear
    y_imp, _ = ddsp_vmap_tubescreamer_noninv_process(impulse, state, params)

    Y = np.fft.rfft(np.array(y_imp))
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(Y) + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.plot(freq, mag_db)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("ddsp_vmap_tubescreamer_noninv – Small-signal Frequency Response")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Audio demo
    # -------------------------------------------------------------------------

    try:
        print("=== Audio Demo: input.wav ===")
        x_audio, fs_in = sf.read("input.wav", dtype="float32")
        if fs_in != fs:
            print(f"Warning: expected fs={fs}, got {fs_in}. Playing at fs_in.")

        if x_audio.ndim > 1:
            x_audio = x_audio[:, 0]

        x_jax = jnp.array(x_audio, dtype=jnp.float32)
        state_audio = state

        y_audio, _ = ddsp_vmap_tubescreamer_noninv_process(x_jax, state_audio, params)

        # Normalize for safety
        y_audio = y_audio / (jnp.max(jnp.abs(y_audio)) + 1e-6)
        y_np = np.array(y_audio)

        print("Playing dry...")
        sd.play(x_audio, fs_in)
        sd.wait()

        print("Playing wet...")
        sd.play(y_np, fs_in)
        sd.wait()

    except Exception as e:
        print("Audio demo skipped:", e)
```

```
