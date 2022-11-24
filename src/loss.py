import tensorflow as tf

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
    reference_stft = convert_stft_from_amplitude_phase(y_true)
    estimation_stft = convert_stft_from_amplitude_phase(y_pred)
    loss = tf.keras.losses.mean_squared_error(reference_stft, estimation_stft)
    if train:
        return loss
    else:
        # For metric
        loss = tf.cast(loss, dtype=tf.float32)
        loss = tf.math.reduce_mean(loss)
        return loss


def mean_absolute_error_amplitdue_phase(y_true, y_pred, train=True):
    reference_stft = convert_stft_from_amplitude_phase(y_true)
    estimation_stft = convert_stft_from_amplitude_phase(y_pred)
    loss = tf.keras.losses.mean_absolute_error(reference_stft, estimation_stft)
    if train:
        return loss
    else:
        # For metric
        loss = tf.math.reduce_mean(loss)
        return loss


def ideal_amplitude_mask(y_true, y_pred, train=True):
    reference_amplitude = y_true[..., 0, :, :, :]
    estimation_amplitude = y_pred[..., 0, :, :, :]

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
    reference_amplitude = y_true[..., 0, :, :, :]
    reference_phase = y_true[..., 1, :, :, :]
    estimation_amplitude = y_pred[..., 0, :, :, :]
    estimation_phase = y_pred[..., 1, :, :, :]

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
    """[TODO] After backpropagation, evaluation is not nan, but when training it goes to nan
    Loss = norm_2(|X|^0.3-[X_bar|^0.3) + 0.113*norm_2(X^0.3-X_bar^0.3)

    Q. How complex number can be power 0.3?
      x + yi = r*e^{jtheta}
      (x + yi)*0.3 = r^0.3*e^{j*theta*0.3}


      X^0.3-X_bar^0.3 r^{0.3}*e^{j*theta*0.3} - r_bar^{0.3}*e^{j*theta_bar*0.3}
    """
    reference_amplitude = tf.cast(y_true[..., 0, :, :, :], dtype=tf.complex64)
    reference_phase = tf.cast(y_true[..., 1, :, :, :], dtype=tf.complex64)
    estimation_amplitude = tf.cast(y_pred[..., 0, :, :, :], dtype=tf.complex64)
    estimation_phase = tf.cast(y_pred[..., 1, :, :, :], dtype=tf.complex64)

    loss_absolute = tf.math.pow(
        tf.math.pow(reference_amplitude, 0.3) - tf.math.pow(estimation_amplitude, 0.3),
        2,
    )
    loss_phase = 0.113 * tf.math.pow(
        tf.math.pow(reference_amplitude, 0.3) * tf.math.exp(1j * reference_phase * 0.3)
        - tf.math.pow(estimation_amplitude, 0.3)
        * tf.math.exp(1j * estimation_phase * 0.3),
        2,
    )
    loss = loss_absolute + loss_phase
    if train:
        return loss
    else:
        # For metric
        loss = tf.math.reduce_mean(loss)
        return loss