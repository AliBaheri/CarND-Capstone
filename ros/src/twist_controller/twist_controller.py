GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED_YAW_CONTROLLER = 1.0

# Import helper classes
from  yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID
import rospy
import math


class Controller(object):
	def __init__(self, *args, **kwargs):
		self.yaw_controller = YawController(kwargs['wheel_base'], kwargs['steer_ratio'],
											MIN_SPEED_YAW_CONTROLLER, kwargs['max_lat_accel'],
											kwargs['max_steer_angle'])
		self.throttle_pid = PID(kp=0.1, ki=0.02, kd=0.0, mn=kwargs['decel_limit'], mx=kwargs['accel_limit'])
		self.prev_time = None
		self.brake_deadband = kwargs['brake_deadband']
		self.wheel_radius = kwargs['wheel_radius']
		self.vehicle_mass = kwargs['vehicle_mass']

	def control(self, *args, **kwargs):
		target_velocity_linear_x = args[0]
		target_velocity_angular_z = args[1]
		current_velocity_linear_x = args[2]
		current_velocity_angular_z = args[3]
		dbw_enabled = args[4]
		throttle = 0.0
		brake = 0.0

		if not dbw_enabled:
			self.throttle.reset()
			return 0, 0, 0

		# Compute difference between target and current velocity as CTE for throttle.
		diff_velocity = target_velocity_linear_x - current_velocity_linear_x

		current_time = rospy.get_time()
		dt = 0
		if self.prev_time is not None:
			dt = (current_time - self.prev_time).to_sec() # Should already be in second, just make sure it is
		self.prev_time = current_time

		velocity_controller = 0
		if dt > 0:
			velocity_controller = self.throttle_pid.step(diff_velocity, dt)

		if velocity_controller >= 0:
			throttle = velocity_controller
			brake = 0
		elif velocity_controller <= self.brake_deadband:
			brake = -velocity_controller * self.vehicle_mass * self.wheel_radius
		else:
			brake = 0
			throttle = 0

		steering = self.yaw_controller.get_steering(target_velocity_linear_x, target_velocity_angular_z, current_velocity_linear_x)

		self.prev_time = current_time

		return throttle, brake, steering
