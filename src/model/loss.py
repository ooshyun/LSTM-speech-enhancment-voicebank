import tensorflow as tf
from keras.backend import epsilon
def convert_stft_from_amplitude_phase(y):
    y_amplitude = y[..., 0, :, :, :]  # amp/phase, ch, frame, freq
    y_phase = y[..., 1, :, :, :]
    y_amplitude = tf.cast(y_amplitude, dtype=tf.complex64)
    y_phase = tf.math.multiply(
        tf.cast(1j, dtype=tf.complex64), tf.cast(y_phase, dtype=tf.complex64)
    )

    return tf.math.multiply(y_amplitude, tf.math.exp(y_phase))


def convert_stft_from_real_imag(y):
    y_real = y[..., 0, :, :, :]  # amp/phase, ch, frame, freq
    y_imag = y[..., 1, :, :, :]
    y_real = tf.cast(y_real, dtype=tf.complex64)
    y_imag = tf.math.multiply(
        tf.cast(1j, dtype=tf.complex64), tf.cast(y_imag, dtype=tf.complex64)
    )

    return tf.add(y_real, y_imag)


def mean_square_error_amplitdue_phase(y_true, y_pred, train=True):
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    if train:
        return loss
    else:
        # For metric
        loss = tf.cast(loss, dtype=tf.float32)
        loss = tf.math.reduce_mean(loss)
        return loss


def mean_absolute_error_amplitdue_phase(y_true, y_pred, train=True):
    loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    if train:
        return loss
    else:
        # For metric
        loss = tf.math.reduce_mean(loss)
        return loss


def ideal_amplitude_mask(y_true, y_pred, train=True):
    reference_amplitude = tf.abs(y_true)
    estimation_amplitude = tf.abs(y_pred)

    loss = tf.keras.losses.mean_absolute_error(reference_amplitude, estimation_amplitude)
    if train:
        return loss
    else:
        # For metric
        loss = tf.cast(loss, dtype=tf.float32)
        loss = tf.math.reduce_mean(loss)
        return loss

def phase_sensitive_spectral_approximation_loss(y_true, y_pred, train=True):
    """
    D_psa(mask) = (mask|y| - |s|cos(theta))^2
    theta = theta_s - theta_y
    """
    epsilon_tensor = tf.complex(real=tf.zeros_like(y_true, dtype=tf.float32), imag=tf.ones_like(y_true, dtype=tf.float32)*epsilon())
    reference_amplitude = tf.abs(y_true)
    reference_phase = tf.math.angle(y_true+epsilon_tensor)
    estimation_amplitude = tf.abs(y_pred)
    estimation_phase = tf.math.angle(y_pred+epsilon_tensor)

    reference_amplitude = tf.math.multiply(
        reference_amplitude, tf.math.cos(estimation_phase - reference_phase)
    )

    loss = tf.keras.losses.mean_absolute_error(reference_amplitude, estimation_amplitude)
    if train:
        return loss
    else:
        # For metric
        loss = tf.math.reduce_mean(loss)
        return loss


def phase_sensitive_spectral_approximation_loss_bose(y_true, y_pred, train=True):
    """
    Loss = norm_2(|X|^0.3-[X_bar|^0.3) + 0.113*norm_2(X^0.3-X_bar^0.3)

    Q. How complex number can be power 0.3?
      x + yi = r*e^{jtheta}
      (x + yi)*0.3 = r^0.3*e^{j*theta*0.3}

      X^0.3-X_bar^0.3 r^{0.3}*e^{j*theta*0.3} - r_bar^{0.3}*e^{j*theta_bar*0.3}
    """
    epsilon_tensor = tf.complex(real=tf.zeros_like(y_true, dtype=tf.float32), imag=tf.ones_like(y_true, dtype=tf.float32)*epsilon())
    reference_amplitude = tf.abs(y_true) + epsilon()
    reference_phase = tf.math.angle(y_true+epsilon_tensor)
    estimation_amplitude = tf.abs(y_pred) + epsilon()
    estimation_phase = tf.math.angle(y_pred+epsilon_tensor)

    loss_absolute = tf.math.pow(
        tf.math.pow(reference_amplitude, 0.3) - tf.math.pow(estimation_amplitude, 0.3),
        2,
    )

    reference_amplitude_power = tf.cast(tf.math.pow(reference_amplitude, 0.3), dtype=tf.complex64)
    reference_phase_power = tf.complex(real=tf.zeros_like(reference_phase, dtype=tf.float32), imag=reference_phase * 0.3)
    estimation_amplitude_power = tf.cast(tf.math.pow(estimation_amplitude, 0.3), dtype=tf.complex64)
    estimation_phase_power = tf.complex(real=tf.zeros_like(estimation_phase, dtype=tf.float32), imag=estimation_phase * 0.3)
    
    diff_power_law_compressed = reference_amplitude_power*reference_phase_power - estimation_amplitude_power*estimation_phase_power
    const_power_two = tf.cast(tf.ones(1, dtype=tf.float32)*2, dtype=tf.complex64)
    
    loss_phase = 0.113 * tf.pow(diff_power_law_compressed, const_power_two)
    loss_phase = tf.math.real(loss_phase)
    loss = loss_absolute + loss_phase
    if train:
        return loss
    else:
        # For metric
        loss = tf.math.reduce_mean(loss)
        return loss