GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_TORQUE= 20000

# Import helper classes
from  yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID
import rospy
import math


class Controller(object):
	def __init__(self, *args, **kwargs):
		self.yaw_controller = YawController(kwargs['wheel_base'], kwargs['steer_ratio'],
											kwargs['min_speed'] + ONE_MPH, kwargs['max_lat_accel'],
											kwargs['max_steer_angle'])
		self.steering_controller = PID(0.5, 0.05, 0.1, -0.35, 0.35)
		self.min_speed = kwargs['min_speed']
		self.prev_time = rospy.get_time()
		self.brake_deadband = kwargs['brake_deadband']
		self.total_mass = kwargs['vehicle_mass'] + kwargs['fuel_capacity']*GAS_DENSITY
		self.wheel_radius = kwargs['wheel_radius']
		self.accel_limit = kwargs['accel_limit']
		self.decel_limit = kwargs['decel_limit']
		
		

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
		dt = current_time - self.prev_time
		self.prev_time = current_time

		#velocity_controller = 0
		#if dt > 0:
		#	velocity_controller = self.throttle_pid.step(diff_velocity, dt)

		#if velocity_controller > 0:
            	#	throttle = velocity_controller
            	#	brake = 0
        	#else:
            	#	throttle = 0
            	#	decel = -velocity_controller

            	#	if decel < self.brake_deadband:
                #		decel= 0

            	#	brake = decel * self.total_mass *self.wheel_radius				
		corrective_steer = self.steering_controller.step(target_velocity_angular_z, dt)
		yaw_steer = self.yaw_controller.get_steering(target_velocity_linear_x, target_velocity_angular_z, current_velocity_linear_x)
		steering = corrective_steer + yaw_steer
		accel_time = 0.5
        	acceleration = diff_velocity / accel_time
	        if acceleration > 0:
        	    acceleration = min(self.accel_limit, acceleration)
        	else:
        	    acceleration = max(self.decel_limit, acceleration)
        	
	        torque = self.total_mass * acceleration * self.wheel_radius
        	throttle, brake = 0, 0
        	if torque > 0:
        	    
        	    throttle, brake = min(1.0, torque/10.0), 0.0
	        else:
	            throttle, brake = 0.0, min(abs(torque),MAX_TORQUE)

		
		return throttle, brake, steering	
