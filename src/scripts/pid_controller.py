class PIDController:

    """ PIDController class implements a Proportional-Integral-Derivative controller.

    Args:

        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        setpoint (float): Setpoint for the controller.

    Attributes:

        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        setpoint (float): Setpoint for the controller.
        clip_value (function): Lambda function for clipping values within a range.
        error (float): Current error.
        last_error (float): Error from the previous step.
        integral (float): Integral term.
        derivative (float): Derivative term.

    Methods:

        update(measured_value): Update the controller with the measured value. """

    def __init__(self, kp, ki, kd, setpoint, output_limit, integral_limit):

        self.kp         = kp        # Proportional gain
        self.ki         = ki        # Integral gain
        self.kd         = kd        # Derivative gain
        self.setpoint   = setpoint  # Setpoint for the controller

        self.output_limit   = output_limit
        self.integral_limit = integral_limit

        self.clip_value = lambda x, lower, upper: lower if x < lower else upper if x > upper else x

        self.error      = 0.0  # Current error
        self.last_error = 0.0  # Error from the previous step
        self.integral   = 0.0  # Integral term
        self.derivative = 0.0  # Derivative term

    def update(self, measured_value):

        """ Update the controller with the measured value and time step.

        Args:
            measured_value (float): Measured value from the system.

        Returns:
            float: Control output. """

        self.error =  measured_value - self.setpoint
        # Proportional term
        p_term = self.kp * self.error

        # Integral term
        self.integral += self.error
        i_term = self.ki * self.integral
        i_term = self.clip_value(i_term, 
                                self.integral_limit[0], 
                                self.integral_limit[1])

        # Derivative term
        self.derivative = (self.error - self.last_error) 
        d_term = self.kd * self.derivative

        # Compute the control output
        control_output = p_term + i_term + d_term
        control_output = self.clip_value(control_output, 
                                         self.output_limit[0], 
                                         self.output_limit[1])

        # Update the last error for the next step
        self.last_error = self.error

        return control_output

