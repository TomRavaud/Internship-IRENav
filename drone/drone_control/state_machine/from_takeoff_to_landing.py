#!/usr/bin/env python3

import rospy
from smach import State, StateMachine
from smach_ros import ServiceState, SimpleActionState
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest, CommandTOL, CommandTOLRequest
from drone_control.msg import IBVSAction, IBVSGoal
import cv2
import cv_bridge

# class TakeOff(State):
#     def __init__(self, altitude):
#         State.__init__(self, outcomes=["takeoff_succeeded"])
#         self.altitude = altitude
    
#     def execute(self, ud):
#         # Set the flight mode to guided
#         mode = SetModeRequest()
#         mode.custom_mode = "GUIDED"
#         set_mode_client.call(mode)
    
#         # Arm the throttles
#         arming = CommandBoolRequest()
#         arming.value = True
#         arming_client.call(arming)
        
#         # Take-off
#         takeoff = CommandTOLRequest()
#         takeoff.altitude = self.altitude
#         takeoff_client.call(takeoff)
        
#         rospy.Rate(0.2).sleep()
        
#         return 'takeoff_succeeded'
    
# class Land(State):
#     def __init__(self):
#         State.__init__(self, outcomes=["landing_succeeded"])
    
#     def execute(self, ud):
#         # Set the flight mode to land
#         mode = SetModeRequest()
#         mode.custom_mode = "LAND"
#         set_mode_client.call(mode)
        
#         return 'landing_succeeded'
    
# class Stabilize(State):
#     def __init(self):
#         return
    
#     def execute(self, ud):
#         return 'stabilization_succeeded'

class Wait(State):
    def __init__(self, duration):
        State.__init__(self, outcomes=["succeeded_waiting"])
        self.duration = duration
        
    def execute(self, ud):
        rospy.Rate(1./self.duration).sleep()
        return "succeeded_waiting"

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("from_takeoff_to_landing")
    
    from_takeoff_to_landing = StateMachine(outcomes=["succeeded_sm", "aborted_sm", "preempted_sm"]) 
    
    with from_takeoff_to_landing:
        mode_guided = SetModeRequest()
        mode_guided.custom_mode = "GUIDED"
        
        StateMachine.add("MODE_GUIDED", ServiceState("/mavros/set_mode",
                                                     SetMode,
                                                     request=mode_guided),
                        transitions={"succeeded":"ARMING",
                                     "aborted":"aborted_sm",
                                     "preempted":"preempted_sm"})
        
        arming = CommandBoolRequest()
        arming.value = True
        
        StateMachine.add("ARMING",
                         ServiceState("mavros/cmd/arming",
                                      CommandBool,
                                      request=arming),
                        transitions={"succeeded":"TAKEOFF",
                                     "aborted":"aborted_sm",
                                     "preempted":"preempted_sm"})
        TAKEOFF_ALTITUDE = 4
        takeoff = CommandTOLRequest()
        takeoff.altitude = TAKEOFF_ALTITUDE
        
        StateMachine.add("TAKEOFF",
                         ServiceState("mavros/cmd/takeoff",
                                      CommandTOL,
                                      request=takeoff),
                         transitions={"succeeded":"WAITING",
                                      "aborted":"aborted_sm",
                                      "preempted":"preempted_sm"})
        
        WAITING_TIME = 10
        
        StateMachine.add("WAITING",
                         Wait(WAITING_TIME),
                         transitions={"succeeded_waiting":"STABILIZING"})
        
        target_image_cv = cv2.imread(
            "/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/target_image_1475_wall.png",
            flags=cv2.IMREAD_GRAYSCALE)
        
        bridge = cv_bridge.CvBridge()
        
        IBVS_goal = IBVSGoal()
        # IBVS_goal.stabilize = True
        IBVS_goal.target_image = bridge.cv2_to_imgmsg(target_image_cv, encoding="passthrough")
        
        StateMachine.add("STABILIZING",
                         SimpleActionState("drone/IBVS",
                                           IBVSAction,
                                           goal=IBVS_goal),
                         transitions={"succeeded":"MODE_LAND",
                                      "aborted":"aborted_sm",
                                      "preempted":"preempted_sm"})
        
        # mode_land = SetModeRequest()
        # mode_land.custom_mode = "LAND"
        
        # StateMachine.add("MODE_LAND",
        #                  ServiceState("mavros/set_mode",
        #                               SetMode,
        #                               request=mode_land),
        #                  transitions={"succeeded":"succeeded_sm",
        #                               "aborted":"aborted_sm",
        #                               "preempted":"preempted_sm"})
        
    from_takeoff_to_landing.execute() 
