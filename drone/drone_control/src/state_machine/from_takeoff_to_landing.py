#!/usr/bin/env python3

import rospy

# State machines libraries
from smach import State, StateMachine, Concurrence
from smach_ros import ServiceState, SimpleActionState, IntrospectionServer

# Import mavros services
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest, CommandTOL, CommandTOLRequest

# Import some custom messages
from drone_control.msg import IBVSAction, IBVSGoal, PoseEstimationAction, PoseEstimationGoal, DecisionAction, DecisionGoal

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

    
def preempt_remaining_running_states(outcome_map):
    """Function called when the first concurrent state terminates. It sends a
    preempt request to remaining running concurrent states

    Args:
        outcome_map (dictionary): Mapping between states and their outcomes

    Returns:
        bool: If true, sends a preempt request to remaining running states
        (if False, tells the state machine to keep running)
    """
    return True
    
def out_concurrence(outcome_map):
    """Function called when the last concurrent state terminates. It decides
    of the outcome of the Concurrence container

    Args:
        outcome_map (dictionary): Mapping between states and their outcomes

    Returns:
        string: Concurrence container outcome
    """
    if outcome_map["DECIDING"] == "succeeded":
        return "ready_to_land"
    return "aborted_concurrence"
    

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("from_takeoff_to_landing")
    
    # Create the top level state machine
    from_takeoff_to_landing = StateMachine(outcomes=["succeeded_sm", "aborted_sm", "preempted_sm"]) 
    
    # Open the StateMachine container
    with from_takeoff_to_landing:
        
        ### Service state : GUIDED ###
        mode_guided = SetModeRequest()
        mode_guided.custom_mode = "GUIDED"
        
        StateMachine.add("MODE_GUIDED", ServiceState("/mavros/set_mode",
                                                     SetMode,
                                                     request=mode_guided),
                        transitions={"succeeded":"ARMING",
                                     "aborted":"aborted_sm",
                                     "preempted":"preempted_sm"})
        
        ### Service state : ARMING ###
        arming = CommandBoolRequest()
        arming.value = True
        
        StateMachine.add("ARMING",
                         ServiceState("mavros/cmd/arming",
                                      CommandBool,
                                      request=arming),
                        transitions={"succeeded":"TAKEOFF",
                                     "aborted":"aborted_sm",
                                     "preempted":"preempted_sm"})
        
        ### Service state : TAKEOFF ###
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
        
        ### State : WAITING (for the take-off to finish) ###
        WAITING_TIME = 10
        
        StateMachine.add("WAITING",
                         Wait(WAITING_TIME),
                         transitions={"succeeded_waiting":"CONCURRENCE"})
        
        
        # Create a sub state machine (a concurrent state machine)
        stabilize_and_decide = Concurrence(outcomes=["ready_to_land", "aborted_concurrence"],
                                           default_outcome="aborted_concurrence",
                                           child_termination_cb=preempt_remaining_running_states,
                                           outcome_cb=out_concurrence)
        # stabilize_and_decide = Concurrence(outcomes=["ready_to_land", "aborted_concurrence"],
        #                                    default_outcome="aborted_concurrence",
        #                                    outcome_map={"ready_to_land":{"DECIDING":"succeeded"}})
        
        # Open the Concurrence container
        with stabilize_and_decide:
            
            ### Concurrent Action state : STABILIZING ###
            # Set the target image of IBVS
            target_image_cv = cv2.imread(
                "/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/Images/target_image_1475_wall_cropped.png",
                flags=cv2.IMREAD_GRAYSCALE)

            # Use to convert OpenCV format to ROS Image
            bridge = cv_bridge.CvBridge()
            
            # Make this image the IBVS action's goal
            IBVS_goal = IBVSGoal()
            IBVS_goal.target_image = bridge.cv2_to_imgmsg(target_image_cv, encoding="passthrough")

            # Concurrence.add("STABILIZING",
            #                  SimpleActionState("drone/IBVS",
            #                                    IBVSAction,
            #                                    goal=IBVS_goal))
            
            ### Concurrent Action state : ESTIMATING_POSE ###
            # The goal is not important here
            pose_estimation_goal = PoseEstimationGoal()
            pose_estimation_goal.estimating_pose = True
            
            Concurrence.add("ESTIMATING_POSE",
                            SimpleActionState("drone/PoseEstimation",
                                              PoseEstimationAction,
                                              goal=pose_estimation_goal))
            
            ### Concurrent Action State : DECIDING ###
            decision_goal = DecisionGoal()
            decision_goal.mode = "reactive"
            
            Concurrence.add("DECIDING",
                            SimpleActionState("drone/Decision",
                                              DecisionAction,
                                              goal=decision_goal))
            
        StateMachine.add("CONCURRENCE", stabilize_and_decide,
                         transitions={"ready_to_land":"succeeded_sm",
                                      "aborted_concurrence":"aborted_sm"})

        mode_land = SetModeRequest()
        mode_land.custom_mode = "LAND"
        
        # StateMachine.add("MODE_LAND",
        #                  ServiceState("mavros/set_mode",
        #                               SetMode,
        #                               request=mode_land),
        #                  transitions={"succeeded":"succeeded_sm",
        #                               "aborted":"aborted_sm",
        #                               "preempted":"preempted_sm"})
    
    # Instantiate an introspection server which allows us to display
    # the state machine structure and helps us to debug it
    sis = IntrospectionServer("sis", from_takeoff_to_landing,
                              "/FROM_TAKEOFF_TO_LANDING")
    
    sis.start()
    
    # Execute the main state machine
    from_takeoff_to_landing.execute()
    
    sis.stop()
