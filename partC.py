
#
# === Introduction ===
#
# In this problem, you will again build a planner that helps a robot
#   find the best path through a warehouse filled with boxes
#   that it has to pick up and deliver to a dropzone. Unlike Project 2- 
#   Part B, however, in this problem the robot's movement is subject to error.
#   Moreover, the robot will not have access to a plan of the warehouse
#   beforehand, and the robot's perception of its surrounding will also
#   be subject to error.
# 
#
# === Input Specifications ===
# 
# The specifications of the warehouse are as in Part B, but the warehouse layout
#   will not be available at the beginning of the task. Instead, the robot will
#   perceive its surroundings by being given data through the `process_measurement`
#   function.
#
# In this part of the project, the order in which the boxes are delivered does not
#   matter. `todo` will simply be an integer >= 1 giving the total number of boxes
#   that must be delivered.
# 
# === Rules for Measurement ===
#
# Each turn, the `process_measurement` method will be called before `next_move`. 
#   You should use the measurement data to update your robot's belief of its 
#   current position before planning your next move.
#
# The data will consist of tuples of distances and relative bearings to markers 
#   within line-of-sight of the robot's center point. Markers will be present on 
#   walls and boxes. An example of measurement data may be:
#   ('wall', 'f48e33926568ab51945aa93714d92607', 2.348, 0.913)
#   which means that the robot's center point is distance 2.348 away from wall marker 'f48e33926568ab51945aa93714d92607', 
#   and the bearing from the robot to the marker is 0.913 relative to the robot's current 
#   bearing (so a relative bearing of 0 means the robot would reach the marker by moving 
#   in a straight line at its current bearing)
#
# The distance and relative bearing measurements will be subject to Gaussian noise
#   with mean 0 and with variance increasing as the distance between the robot and the
#   marker increases.
#
# The robot will also receive a boolean that is True if it's currently holding a box and False
#   if not. This boolean will always be correct.
#
# === Rules for Movement ===
#
# - The robot may move any distance between 0 and `max_distance` per time step.
# - The robot may set its steering angle anywhere between -`max_steering` and 
#   `max_steering` per time step. A steering angle of 0 means that the robot will
#   move according to its current bearing. A positive angle means the robot will 
#   turn counterclockwise by `steering_angle` radians; a negative steering_angle 
#   means the robot will turn clockwise by abs(steering_angle) radians.
# - Upon a movement, the robot will change its steering angle instantaneously to the 
#   amount indicated by the move, and then it will move a distance in a straight line in its
#   new bearing according to the amount indicated move.
#   Movements will be  subject to small amounts of noise in the distance (0.99 to 1.01 times the desired
#   distance, never exceeding max_distance) and steering (+/- 0.01 of the desired steering,
#   never violating the constraint that abs(steering) <= max_steering).
# - The cost per turn is 1 plus the amount of distance traversed by the robot on that turn.
#
# - The robot may pick up a box whose center point is within 0.5 units of the robot's center point.
# - If the robot picks up a box, it incurs a total cost of 2 for that turn (this already includes 
#   the 1-per-turn cost incurred by the robot).
# - While holding a box, the robot may not pick up another box.
# - The robot may attempt to put a box down at a total cost of 1.5 for that turn.
#   The box must be placed so that:
#   - The box is not contacting any walls, the exterior of the warehouse, any other boxes, or the robot
#   - The box's center point is within 0.5 units of the robot's center point
# - A box is always oriented so that two of its edges are horizontal and the other two are vertical.
# - If a box is placed entirely within the '@' space, it is considered delivered and is removed from the 
#   warehouse.
# - The warehouse will be arranged so that it is always possible for the robot to move to any 
#   box without having to rearrange any other boxes.
#
# - If the robot crashes, it will stop moving and incur a cost of 100*distance, where distance
#   is the length it attempted to move that turn. (The regular movement cost will not apply.)
# - If an illegal move is attempted, the robot will not move, but the standard cost will be incurred.
#   Illegal moves include (but are not necessarily limited to):
#     - picking up a box that doesn't exist or is too far away
#     - picking up a box while already holding one
#     - putting down a box too far away or so that it's touching a wall, the warehouse exterior, 
#       another box, or the robot
#     - putting down a box while not holding a box
#
# === Output Specifications ===
#
# `next_move` should return a string in one of the following formats:
#
# 'move {steering} {distance}', where '{steering}' is a floating-point number between
#   -`max_steering` and `max_steering` (inclusive) and '{distance}' is a floating-point
#   number between 0 and `max_distance`
# 
# 'lift {bearing}', where '{bearing}' is replaced by the bearing of the box's center point
#   relative to the robot (with 0 meaning right in front of the robot).
#   If a box's center point is within 0.5 units of the robot's center point and its bearing
#   is within 0.2 radians of the specified bearing, then the box is lifted. Otherwise, nothing
#   happens.
#
# 'down {bearing}', where '{bearing}' is replaced by the relative bearing where the
#   robot wishes to place its box. The robot will attempt to place the box so that its center point
#   is at this bearing at a distance of 0.5 units away. If the box can be so placed without intersecting
#   anything, it does so. Otherwise, the placement fails and the robot continues to hold the box.

from operator import itemgetter
import copy
import numpy as np
import random
import math
from math import *
import robot
from collections import OrderedDict # deal with landmarks
PI = math.pi

random.seed(2)

class OnlineDeliveryPlanner:

    def __init__(self, todo, max_distance, max_steering, verbose=False):

        ######################################################################################
        # TODO: You may add to this function for any initialization required for your planner  
        ######################################################################################
        self.todo = todo   #Integer number of boxes in the warehouse 
        self.max_distance = max_distance   #Maximum movement per move/turn
        self.max_steering = max_steering   #Maximum steering angle per move/turn
	self.verbose = verbose   #Optional flag you can toggle from the testing suite.
	self.wallDict=dict()
	self.boxDict =dict()
	self.edgeDict = dict()

	self.LMDict=OrderedDict()
	self.LMCoorDict=OrderedDict()

	self.alpha = 1.0
	self.alpahZeroCount = 0
	self.alphaChanged =False


        #self.calcLM_Num_Edge = self.calcNumOfLandMarks
	


	self.count = 0
	self.numOfLM=0
	self.expandMatNum=0
	
        self.boxCentList=[]
	
	self.initRobotCoor = (0.,0.)
	#self.initOmegaMatrix()
	self.robot = robot.Robot(0., 0., bearing=0.0, 
                    max_distance=max_distance, max_steering=max_steering)
        self.LMOrderNum=0
        self.AllLMIDList=[]

        self.robotCoorList = []
        self.motionList = []
        
        self.robotCoorList.append(self.initRobotCoor)

        self.EdgeNotEnough =True

        self.boxes = dict()
        self.obstacles = []
        self.dropzone = dict()

        self.boxShowed = False
        self.holdBox = False
        self.taskClear = False

        self.boxDeliveredList=[]
        self.prev_boarder=[]

        self.gridNum = 5
        self.boxDelivered =0
        self.maxStep = 10
        self.currStep = 0

        self.mList=[(1.5807963267948966, 0),(0.2203038324910418, 1.567256121989858),(0.,0.),(-1.5807963267948966, 0),(-1.3384375536519686, 1.3260997727400514),(-1.040665209181538, 0.0)]
    
        
        self.NewID  = ''
        self.isBoxMap = False
        self.randomMotionList=[]

        self.ActOnBox = False
        self.boxGuidedPath = False
        self.lastMove = False
        self.emptyPath =False

        self.discardDist = 2.5
        
        self.ROBOT_RADIUS = 0.3

        self.robot_has_box = None

        self.LiftDownFailure = 0

        self.robot_is_crashed = False


    #-------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#
    def init_para_after_gettingMap(self):
        self.BOX_SIZE = 0.1
        self.BOX_DIAG = 0.1414
        
        self.OBSTACLE_DIAG = 1.414
        self.boxes = dict()
        self.obstacles = []
        self.dropzone = dict()


        #for test
        #self.NSWE =[4.5,-0.5,-4.5,0.5]
        
    
        if self.EdgeNotEnough ==False:   
            self.warehouse_limits = {'max_x':self.NSWE[3],'min_x':self.NSWE[2], 'min_y':self.NSWE[1],'max_y':self.NSWE[0]}
            self.warehouse_limits['segments'] = []
            # West segment (x0,y0) -> (x1,y1)
            self.warehouse_limits['segments'].append( 
                        [( self.ROBOT_RADIUS + self.warehouse_limits['min_x'], self.warehouse_limits['min_y']),
                         ( self.ROBOT_RADIUS + self.warehouse_limits['min_x'], self.warehouse_limits['max_y']) ] )
            # South segment
            self.warehouse_limits['segments'].append( 
                        [( self.warehouse_limits['max_x'],self.warehouse_limits['min_y'] + self.ROBOT_RADIUS ),
                         ( self.warehouse_limits['min_x'], self.warehouse_limits['min_y'] + self.ROBOT_RADIUS) ] )
            # East segment
            self.warehouse_limits['segments'].append( 
                        [( self.warehouse_limits['max_x'] - self.ROBOT_RADIUS, self.warehouse_limits['min_y'] ),
                         ( self.warehouse_limits['max_x'] - self.ROBOT_RADIUS, self.warehouse_limits['max_y'] ) ] )
            # North segment
            self.warehouse_limits['segments'].append( 
                        [( self.warehouse_limits['max_x'], self.warehouse_limits['max_y']-self.ROBOT_RADIUS) ,
                         ( self.warehouse_limits['min_x'], self.warehouse_limits['max_y']-self.ROBOT_RADIUS) ] )
        
        
        #print 'before wallCentList'
        for wall in self.wallCentList:
            x = wall[0]-0.5
            y = wall[1]+0.5
            
            obstacle = dict()
            # obstacle edges
            obstacle['min_x'] = x
            obstacle['max_x'] = x + 1.0
            obstacle['min_y'] = y - 1.0
            obstacle['max_y'] = y
            # center of obstacle
            obstacle['ctr_x'] = x + 0.5
            obstacle['ctr_y'] = y - 0.5
            # compute clearance parameters for robot
            obstacle = self._dilate_obstacle_for_robot( obstacle )
            
            self.obstacles.append( obstacle )
        #print 'self.obstacles',self.obstacles
        #print 'after wallCentList'        
        # set the dropzone limits
        
                
        self.dropCent = [0.,0.]
        x = self.dropCent[0]-0.5
        y = self.dropCent[1]+0.5
        self.dropzone['min_x'] = x
        self.dropzone['max_x'] = x + 1.0
        self.dropzone['min_y'] = y - 1.0
        self.dropzone['max_y'] = y
        self.dropzone['ctr_x'] = x + 0.5
        self.dropzone['ctr_y'] = y - 0.5
        
        
        # from test_suite_partB
        for i in range(len(self.boxCentList)):
            # set up the parameters for processing the box as an obstacle and for 
            # picking it up and setting it down
            box = dict()
            # box edges
            box['min_x'] = self.boxCentList[i][0] - self.BOX_SIZE
            box['max_x'] = self.boxCentList[i][0] + self.BOX_SIZE
            box['min_y'] = self.boxCentList[i][1] - self.BOX_SIZE
            box['max_y'] = self.boxCentList[i][1] + self.BOX_SIZE
            # center of obstacle
            box['ctr_x'] = self.boxCentList[i][0]
            box['ctr_y'] = self.boxCentList[i][1]
            # compute clearance parameters for robot
            box = self._dilate_obstacle_for_robot( box )
            self.boxes[str(i)] = box
        #print 'after BoxCentList'
        #print 'self.boxes',self.boxes

        
    def _is_traversable(self, startCoor, destination, distance,incomplete=False):
        # from test_suite_partB
        #print 'enter _is_traversable'
        
        
        NUDGE_DISTANCE = 0.01
        
        # end points of trajectory
        t1 = destination
        t0 =startCoor
        
        chk_distance = distance
        #print 'chk_distance',chk_distance

        # Is the path too close to any box
        min_distance_to_intercept = chk_distance
        self.robot_is_crashed = False

        
        for box_id, box in self.boxes.iteritems():
            # do a coarse check to see if the center of the obstacle
            # is too far away to be concerned with
            dst = self._distance_point_to_line_segment( (box['ctr_x'], box['ctr_y']), t0, t1 )
            if (dst <= self.BOX_DIAG + self.ROBOT_RADIUS):
                # refine the intercept computation
                dst, intercept_point = self._check_intersection(t0,t1,box)
                if dst < min_distance_to_intercept:
                    min_distance_to_intercept = dst
                    min_intercept_point = intercept_point

        #print 'mindist after box',min_distance_to_intercept
        # Is the path too close to any obstacle
        for o in self.obstacles :
            # do a coarse check to see if the center of the obstacle
            # is too far away to be concerned with
            dst = self._distance_point_to_line_segment( (o['ctr_x'],o['ctr_y']), t0, t1 )
            if (dst <= self.OBSTACLE_DIAG + self.ROBOT_RADIUS):
                # refine the intercept computation
                dst, intercept_point = self._check_intersection(t0,t1,o)
                if dst < min_distance_to_intercept:
                    min_distance_to_intercept = dst 
                    min_intercept_point = intercept_point

       # print 'mindist after wall',min_distance_to_intercept

        if self.EdgeNotEnough ==False:
            #print 'mindist after wall',min_distance_to_intercept
            # Check the edges of the warehouse
            dst, intercept_point = self._check_intersection(t0,t1,self.warehouse_limits)
            if dst < min_distance_to_intercept:
                min_distance_to_intercept = dst 
                #min_intercept_point = intercept_point
        #if edge is incomplete
        if self.EdgeNotEnough ==True:
            #print 'EdgeNotEnough ==True in is_traversable'
            edgeSegList=self.getSegList('warehouse')

            #destination not exceed max or min edge
            
            #print 'self.NSWE',self.NSWE
            if destination[1]>self.NSWE[0] or destination[1]<self.NSWE[1] or destination[0]<self.NSWE[2] or destination[0]>self.NSWE[3]:
                return False
            
            #no collision with warehouse Edge
            collision = False
            #print 'before for edge in edgeSegList:'
            for edge in edgeSegList:
                if robot.compute_distance(edge,destination)<0.4: #0.35=0.25*1.4
                    collision = True
            if collision ==True:
                return False
            
        
        if min_distance_to_intercept < chk_distance :
            #print 'min_distance_to_intercept < chk_dist'
            return False
        return True
 
    # Check if the trajectory intersects with a square obstacle
    def _check_intersection(self,t0,t1,obstacle):
        # from test_suite_partB
        
        min_distance_to_intercept = 1.e6
        min_intercept_point = (0.,0.)

        # check each segment
        for s in obstacle['segments']:
            
            dst, intercept_point =  self._linesegment_intersection( t0, t1, s[0], s[1] )
            if dst < min_distance_to_intercept:
                min_distance_to_intercept = dst 
                min_intercept_point = intercept_point
            #if s == self.warehouse_limits['segments'][2]:
             #   print 'this is eastSeg, minDst=',min_distance_to_intercept
            
            
        # if circular corners are defined - check them
        # circular corners occur when dilating a rectangle with a circle
        if obstacle.has_key('corners') :
            for c in obstacle['corners'] :
                dst, intercept_point =  self._corner_intersection( t0, t1, c )
                if dst < min_distance_to_intercept:
                    min_distance_to_intercept = dst 
                    min_intercept_point = intercept_point 

            
        return min_distance_to_intercept, min_intercept_point

    def _linesegment_intersection( self, p0, p1, q0, q1 ):
        # from test_suite_partB
        
        eps = 1.0e-6
        dst = 1.0e6
        intersection = (0.,0.)

        r = p1[0]-p0[0], p1[1]-p0[1]
        s = q1[0]-q0[0], q1[1]-q0[1]
        qmp = q0[0]-p0[0], q0[1]-p0[1] 

        rxs   = r[0]*s[1] - r[1]*s[0]
        qmpxr = qmp[0]*r[1] - qmp[1]*r[0]

        if abs(rxs) >= eps :

            # check for intersection
            # parametric equations for intersection
            # t = (q - p) x s / (r x s)
            t = (qmp[0]*s[1] - qmp[1]*s[0])/rxs

            # u = (q - p) x r / (r x s)
            u = qmpxr/rxs

            # Note that u and v can be slightly out of this range due to
            # precision issues so we round them 
            if (0.0 <= np.round(t,4) <= 1.0) and (0.0 <= np.round(u,4) <= 1.0) :
                dx, dy = t*r[0], t*r[1]
                dst = math.sqrt( dx*dx + dy*dy)
                intersection =  (p0[0] + dx, p0[1] + dy)

        return dst, intersection
    
    # Find the intersection of a line segment and a semicircle as defined in 
    # the corner dictionary 
    # Use quadratic solution to solve simultaneous equations for
    # (x-a)^2 + (y-b)^2 = r^2 and y = mx + c
    def _corner_intersection( self, t0, t1, corner ):
        # from test_suite_partB
        
        dst = 1.e6
        intercept_point = (0.,0.)

        # Note:  changing nomenclature here so that circle center is a,b
        # and line intercept is c (not b as above)
        a = corner['ctr_x']                 # circle ctrs
        b = corner['ctr_y']
        r = corner['radius']

        # check the case for infinite slope
        dx = t1[0] - t0[0]

        # Find intersection assuming vertical trajectory
        if abs( dx ) < 1.e-6 :
            x0 = t0[0] - a
            #qa = 1.
            qb = -2.*b
            qc = b*b + x0*x0 - r*r
            disc = qb*qb - 4.*qc 

            if disc >= 0.:
                sd = math.sqrt(disc)
                xp = xm = t0[0]
                yp = (-qb + sd)/2.
                ym = (-qb - sd)/2.
      
        # Find intersection assuming non vertical trajectory
        else:
            m = (t1[1] - t0[1])/dx # slope of line
            c = t0[1] - m*t0[0]    # y intercept of line
  
            qa = 1.+m*m
            qb = 2.*(m*c - m*b - a)
            qc = a*a + b*b + c*c - 2.*b*c - r*r

            disc = qb*qb - 4.*qa*qc
  
            if disc >= 0.:
                sd = math.sqrt(disc)
                xp = (-qb + sd) / (2.*qa)
                yp = m*xp + c
                xm = (-qb - sd) / (2.*qa)
                ym = m*xm + c
        
        if disc >= 0. :   
            dp2 = dm2 = 1.e6
            if corner['min_x'] <= xp <= corner['max_x'] and corner['min_y'] <= yp <= corner['max_y'] :
                dp2 = (xp - t0[0])**2 + (yp - t0[1])**2

            if corner['min_x'] <= xm <= corner['max_x'] and corner['min_y'] <= ym <= corner['max_y'] :
                dm2 = (xm - t0[0])**2 + (ym - t0[1])**2

            if dp2 < dm2 :
                # make sure the intersection point is actually on the trajectory segment
                if self._distance_point_to_line_segment( (xp,yp), t0, t1 ) < 1.e-6 :
                    dst = math.sqrt(dp2)
                    intercept_point = (xp, yp)
            else :
                if self._distance_point_to_line_segment( (xm,ym), t0, t1 ) < 1.e-6 :
                    dst = math.sqrt(dm2)
                    intercept_point = (xm, ym)
    

        return dst, intercept_point


    # Find the distance from a point to a line segment
    # This function is used primarily to find the distance between a trajectory
    # segment defined by l0,l1 and the center of an obstacle or box specified
    # by point p.  For a reference see the lecture on SLAM and the segmented CTE
    def _distance_point_to_line_segment( self, p, l0, l1 ):
        # from test_suite_partB
        
        dst = 1.e6
        dx = l1[0] - l0[0]
        dy = l1[1] - l0[1]

        # check that l0,l1 don't describe a point
        d2 = (dx*dx + dy*dy)

        if abs(d2) > 1.e-6:
        
            t = ((p[0] - l0[0]) * dx + (p[1] - l0[1]) * dy)/d2

            # if point is on line segment
            if 0.0 <= t <= 1.0:
                intx, inty = l0[0] + t*dx, l0[1] + t*dy
                dx, dy = p[0] - intx, p[1] - inty
   
            # point is beyond end point
            elif t > 1.0 :
                dx, dy = p[0] - l1[0], p[1] - l1[1]
            
            # point is before beginning point
            else:
                dx, dy = p[0] - l0[0], p[1] - l0[1]
                
            dst = math.sqrt(dx*dx + dy*dy)

        else:
            dx, dy = p[0] - l0[0], p[1] - l0[1]
            dst = math.sqrt(dx*dx + dy*dy)

        return dst


    

    def _dilate_obstacle_for_robot(self, obstacle) :
        # from test_suite_partB
        
        
        # line segments dilated for robot intersection
        obstacle['segments'] = []
        # West segnent
        obstacle['segments'].append( [( obstacle['min_x'] - self.ROBOT_RADIUS, obstacle['max_y'] ),\
                                        ( obstacle['min_x'] - self.ROBOT_RADIUS, obstacle['min_y'] ) ] )
        # South segment
        obstacle['segments'].append( [( obstacle['min_x'], obstacle['min_y'] - self.ROBOT_RADIUS),\
                                        ( obstacle['max_x'], obstacle['min_y'] - self.ROBOT_RADIUS ) ] )
        # East segment
        obstacle['segments'].append( [( obstacle['max_x'] + self.ROBOT_RADIUS, obstacle['min_y'] ),\
                                        ( obstacle['max_x'] + self.ROBOT_RADIUS, obstacle['max_y'] ) ] )
        # North segment
        obstacle['segments'].append( [( obstacle['max_x'], obstacle['max_y'] + self.ROBOT_RADIUS),\
                                        ( obstacle['min_x'], obstacle['max_y'] + self.ROBOT_RADIUS ) ] )

        obstacle['corners'] = []
        # NW corner
        cornerdef = dict()
        cornerdef['ctr_x'] = obstacle['min_x']
        cornerdef['ctr_y'] = obstacle['max_y']
        cornerdef['radius'] = self.ROBOT_RADIUS
        cornerdef['min_x'] = obstacle['min_x'] - self.ROBOT_RADIUS
        cornerdef['max_x'] = obstacle['min_x']
        cornerdef['min_y'] = obstacle['max_y']
        cornerdef['max_y'] = obstacle['max_y'] + self.ROBOT_RADIUS
        obstacle['corners'].append(cornerdef)

        # SW corner
        cornerdef = dict()
        cornerdef['ctr_x'] = obstacle['min_x']
        cornerdef['ctr_y'] = obstacle['min_y']
        cornerdef['radius'] = self.ROBOT_RADIUS
        cornerdef['min_x'] = obstacle['min_x'] - self.ROBOT_RADIUS
        cornerdef['max_x'] = obstacle['min_x']
        cornerdef['min_y'] = obstacle['min_y'] - self.ROBOT_RADIUS
        cornerdef['max_y'] = obstacle['min_y']
        obstacle['corners'].append(cornerdef)

        # SE corner
        cornerdef = dict()
        cornerdef['ctr_x'] = obstacle['max_x']
        cornerdef['ctr_y'] = obstacle['min_y']
        cornerdef['radius'] = self.ROBOT_RADIUS
        cornerdef['min_x'] = obstacle['max_x'] 
        cornerdef['max_x'] = obstacle['max_x'] + self.ROBOT_RADIUS 
        cornerdef['min_y'] = obstacle['min_y'] - self.ROBOT_RADIUS
        cornerdef['max_y'] = obstacle['min_y']
        obstacle['corners'].append(cornerdef)

        # NE corner
        cornerdef = dict()
        cornerdef['ctr_x'] = obstacle['max_x']
        cornerdef['ctr_y'] = obstacle['max_y']
        cornerdef['radius'] = self.ROBOT_RADIUS
        cornerdef['min_x'] = obstacle['max_x'] 
        cornerdef['max_x'] = obstacle['max_x'] + self.ROBOT_RADIUS 
        cornerdef['min_y'] = obstacle['max_y']
        cornerdef['max_y'] = obstacle['max_y'] + self.ROBOT_RADIUS 
        obstacle['corners'].append(cornerdef)

        return obstacle

    def process_measurement(self, data, verbose = False):

        
        
        if True:
            # This is what the measurement data look like
            #print "Obs \tID    Dist    Brg (deg)"

            self.updateLMIDList=[]
            
            for meas in data:
                # label the marker
                #print "{} \t{}     {:6.2f}    {:6.2f}".format(meas[0],meas[1], meas[2],math.degrees(meas[3]))
        ####################################################################################################
                # add new lm to dict
                if meas[2] <self.discardDist:
                    if meas[1] not in self.LMDict:
                        self.expandMatNum += 1
                        self.LMDict[meas[1]]=[meas[0],meas[2],meas[3],self.LMOrderNum,[]]
                        self.LMDict[meas[1]][4].append((meas[2],meas[3]))
                        self.LMOrderNum += 1
                        self.updateLMIDList.append(meas[1])
                        self.AllLMIDList.append(meas[1])

                    else: #update lm info, so order number will be kept
                        #meas[0]:name , [1]:id , [2]:distance, [3]:relative bearing to robot bearing. [4] saved dist&bearing list
                        self.LMDict[meas[1]][4].append((meas[2],meas[3]))
                        self.LMDict[meas[1]]=[meas[0],meas[2],meas[3],self.LMDict[meas[1]][3],self.LMDict[meas[1]][4]]
                        #lmdict[key][0]: name, [1]:distance,[2]:relative bearing to robot bearing.
                        self.updateLMIDList.append(meas[1]) # find out ids which updated
           
            #print 'LMDict[0]',self.LMDict.iteritems().next() #-->get coor

            
            if self.count ==0: # first measurement
                #print 'self.count'
                self.numOfLM = len(self.LMDict)

                # add coor to robotCoorList
                self.robotCoorList.append((0.,0.))
                # [0]:name 1:index 2:coor 3: id no, robot dont have it
                self.LMCoorDict['robot']=['robot',0,(0.,0.)] # first element in LMCOORDICT is robot position.

                
                #landmark part
                self.updateLMinMatrixs()

            else: # not the first measurement
                self.updateLMinMatrixs()                
        else: # no verbose
            pass
                
              
        
        self.count += 1


        
    def updateLMinMatrixs(self):
        self.localizeDict = dict()
        clist=[]
        elist =[]
        
        #print 'self.updateLMIDList length:' ,len(self.updateLMIDList)

        
        for lmID in self.updateLMIDList:                    
            orderNum = self.LMDict[lmID][3] #start from 0
            #print 'orderNum is:',orderNum
            index = orderNum +1
            
            #lmdict[key][0]: name, [1]:distance,[2]:relative bearing to robot bearing.
            dist =  self.LMDict[lmID][1]
            theta = self.robot.bearing +  self.LMDict[lmID][2]

            theta_truncate = robot.truncate_angle(theta)
            
            dx = math.cos(theta) * dist   # lm x - self.robot.x
            dy = math.sin(theta) * dist   # lm y - self.robot.y
            #update it in matrix
            measurement_truncate =(math.cos(theta_truncate) * dist,math.sin(theta_truncate) * dist)

            # key:ID/ 'robot' [0]:name 1:index 2:coor 3:corresponding robot Pos 4: List of Right Coor/robot pos/dist 5: sum of corr,  1/dist
            if lmID not in self.LMCoorDict:
                self.LMCoorDict[lmID]=[self.LMDict[lmID][0],index,(dx,dy),self.LMCoorDict['robot'][2],[],[(0.,0.),0.]]
                #self.LMCoorDict[lmID].append([])
                self.LMCoorDict[lmID][4].append([(dx+self.robot.x,self.robot.y+dy),(self.robot.x,self.robot.y),self.LMDict[lmID][1]])
                self.LMCoorDict[lmID][5] = [(dx+self.robot.x,self.robot.y+dy), 1.0/dist]
                
            else: #already exist
                self.LMCoorDict[lmID][2] = (dx,dy)
                self.LMCoorDict[lmID][3] = self.LMCoorDict['robot'][2]
                self.LMCoorDict[lmID][4].append([(dx+self.robot.x,self.robot.y+dy),(self.robot.x,self.robot.y),self.LMDict[lmID][1]])
                #print 'warehouse data:',self.LMCoorDict[lmID][4]

                #update tuple in [5]
                new_x = self.LMCoorDict[lmID][5][0][0]
                new_x += dx + self.robot.x #update sum x
                new_y = self.LMCoorDict[lmID][5][0][1]
                new_y += dy + self.robot.y #update sum y
                sumW = self.LMCoorDict[lmID][5][1] 
                sumW   += 1.0/dist #sum of weight
                self.LMCoorDict[lmID][5][0]=(new_x,new_y)
                self.LMCoorDict[lmID][5][1]= sumW
                # avg coor of LM: (self.LMCoorDict[lmID][5][0][0]/self.LMCoorDict[lmID][5][1],self.LMCoorDict[lmID][5][0][1]/self.LMCoorDict[lmID][5][1])
            
            #-------------------------update noise----------------------------------
   
            # key:ID   [0]:name wall/warehouse [1]:estimate lm coor
            self.localizeDict[lmID]= [self.LMDict[lmID][0],(dx+self.robot.x,self.robot.y+dy)]
            
            if self.LMDict[lmID][0]=='wall':
                clist.append((dx+self.robot.x,self.robot.y+dy))
            elif self.LMDict[lmID][0]=='warehouse':               
                elist.append((dx+self.robot.x,self.robot.y+dy))

        #print 'clist',clist
        #print 'elist',elist
        
        lastDist = 0.    
        if self.count >0:
            lastDist = robot.compute_distance((self.robot.x,self.robot.y),self.robotCoorList[-1])
        if True:
            #print 'currCoor'
            currCoor = self.LMCoorDict['robot'][2]
            #print 'coorError'
            coorError = self.getWallCentListForLocalizing(currCoor,clist,elist)
            
            #print 'coor noise is:',coorError

            #if max(abs(coorError[0]) ,abs(coorError[1])) > 0.1:
            if max(abs(coorError[0]),abs(coorError[1]))>0.001:
                if lastDist > 0.2: # update dist only when 
                    oldPos = (self.robot.x,self.robot.y)

                    alpha = self.alpha
                    estimate_x = self.robot.x+ coorError[0]
                    estimate_y = self.robot.y+ coorError[1]
                    
                    
                    self.robot.x = self.robot.x * alpha + (1-alpha) *estimate_x
                    self.robot.y = self.robot.y * alpha + (1-alpha) *estimate_y
                    newPos = (self.robot.x,self.robot.y)                
                    prevPos = self.robotCoorList[-1]

                    oldBearing = self.robot.bearing

                    diff_bearing =  robot.compute_bearing(prevPos,newPos)-self.robot.bearing
                    self.robot.bearing = robot.compute_bearing(prevPos,newPos)
                    for lmID in self.updateLMIDList:
                        self.LMCoorDict[lmID][3] = (self.robot.x,self.robot.y) # update robot coor
                    #print 'coor noise is:',coorError,'diff_bearing',diff_bearing,'bearing',self.robot.bearing
            else:
                pass
                #print 'coor noise is too less',coorError
            
        self.robotCoorList.append((self.robot.x,self.robot.y)) # prev: in motionMatrixs

        
        
            



        
    def updateByMotioninMatrixs(self,bearing,dist):

        self.robot.move(bearing,dist)
        self.LMCoorDict['robot'][2] = (self.robot.x,self.robot.y)
        

        


          
    
        
    def next_move(self, verbose = False):

        s = self.getNextMotion()
        return s
        
    def boxGuidedMotion(self):

        self.isBoxMap = True
        #1.randomWalk until sees a box-->2. go to the box -->3. lift the box
        self.buildGridMap()
        self.init_para_after_gettingMap()

        if self.boxShowed == False and self.holdBox ==False: # not seeing a box
            motion = self.randomWalk()
            s = 'move '+str(motion[0])+' '+str(motion[1])      
            self.updateByMotioninMatrixs(motion[0],motion[1])
            return s
        elif self.boxShowed ==True:
            goal = self.boxCentList[0]
            
    def testMotion(self):
        self.buildGridMap()
        self.init_para_after_gettingMap()

        motion = (0.,0.)
        if len(self.mList)>0:
            motion = self.mList[0]
            self.mList.pop(0)
        else:
            return []
        s = 'move '+str(motion[0])+' '+str(motion[1])
        
        self.updateByMotioninMatrixs(motion[0],motion[1])
        
        return s
    def testMotion2(self):
        self.buildGridMap()
        self.init_para_after_gettingMap()

        #only for testing
        start = (-2.0, 0)
        end = (-3.0, 0)

        #print 'self.NSWE',self.NSWE

        motion = self.random4DirWalk()

        s = 'move '+str(motion[0])+' '+str(motion[1])
        
            
        
        self.updateByMotioninMatrixs(motion[0],motion[1])
        
        return s
    
    def getNextMotion(self):
        self.Prev_Path = []        
        self.buildGridMap()
        self.init_para_after_gettingMap()

        self.lastMove = False
        
        maxMove = 400
        randomMove = 50
        
        

        if self.count >= maxMove:
            return []
        #----------------deal with crash or not lifting-----------------------#
	if self.alpahZeroCount>10:
             alphaChanged = False
             self.alpahZeroCount = 0
	
        if self.alphaChanged == True:
            self.alpha = 0.1
            self.alpahZeroCount += 1

        elif self.alphaChanged == False:
            self.alpha = 0.5
            self.alpahZeroCount = 0

        if self.robot_is_crashed == True:
            self.alphaChanged = True
        
        #self.robot_has_box --> 0,1, None, not the same--> randomwalk
        if self.robot_has_box == None and self.holdBox == True:#did not lift or down
            self.alphaChanged = True
            #print 'Lifting fail loop'
            if len(self.boxDeliveredList)>0:
                del self.boxDeliveredList[-1]
            self.LiftDownFailure += 1
            self.holdBox =False
            motion = self.randomWalk()
            
            s = 'move '+str(motion[0])+' '+str(motion[1])
            return s
            

        elif self.robot_has_box != None and self.holdBox == False:
            self.alphaChanged = True
            #print 'Down fail loop'
            self.boxDelivered -= 1
            self.LiftDownFailure += 1
            self.holdBox =True
            motion = self.randomWalk()
            s = 'move '+str(motion[0])+' '+str(motion[1])
            return s
        #-------------------------------------------------------------------
        #list nearest box first than farer box: sorting only after 
        if len(self.boxCentList) >0:
            rangeList = []
            for box in self.boxCentList:
                rangeList.append([robot.compute_distance(box,(self.robot.x,self.robot.y)),box])
            #sort rangeList
            sorted(rangeList, key=itemgetter(1))
            self.boxCentList=[]
            for r in rangeList:
                self.boxCentList.append(r[1])
        #-------------------------------------------------------------------

        
        if self.boxDelivered >= self.todo:
            self.taskClear=True
            
        if self.taskClear==True:
            #print 'Task finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            return []
        if len(self.randomMotionList)>0:
            #print 'enter randomMotionList',self.randomMotionList
            motion = self.randomMotionList[0]
            self.randomMotionList.pop(0)
            s = 'move '+str(motion[0])+' '+str(motion[1])                
            self.updateByMotioninMatrixs(motion[0],motion[1])
            return s

        if self.count <= randomMove:
            #print 'In random walk 4 dir'
            #motion  = self.randomWalk()
            motion = self.random4DirWalk()
            s = 'move '+str(motion[0])+' '+str(motion[1])                
            self.updateByMotioninMatrixs(motion[0],motion[1])
            return s

        if self.EdgeNotEnough == True:
            self.boxGuidedPath = True
            
            self.initRobot=(self.robot.x,self.robot.y,self.robot.bearing)
            if self.holdBox==False:
                goal = (0.,0.)
                if len(self.boxCentList)>0:
                    goal = self.boxCentList[0]
                #print 'GO TO BOX IN LINE: goal is:',goal
                motion,move = self.StraightMoveToBox(goal)
                #print 'motion:',motion
                (self.robot.x,self.robot.y,self.robot.bearing)=self.initRobot                          
                self.updateByMotioninMatrixs(motion[0],motion[1])              
                
                return move
            
            elif self.holdBox==True:
                #print '# return to dropzone'
                goal = (0.,0.)
                motion,move = self.StraightMoveToBox(goal)
                (self.robot.x,self.robot.y,self.robot.bearing)=self.initRobot
                self.updateByMotioninMatrixs(motion[0],motion[1])

                return move

        #--------------------------Edge is enough------------------------#        
        else: # edge info is enough and saw boxes.           
            #print 'enter edge is enough'
            #print 'self.holdBox',self.holdBox
            if self.holdBox == False:
                #move to box if box is in the map
                #print '#move to box if box is in the map'
                self.motion = (0.,0.)
                self.motionList=[]
                self.move = []                
                self.initRobot = (self.robot.x,self.robot.y,self.robot.bearing)
                #print 'initRobot Prop:',self.initRobot
                self.Prev_Path=[]

                #print 'before startLoc'
                startLoc = self.findNearestGrid((self.robot.x,self.robot.y))
                #print 'goal is'

                if len(self.boxCentList) == 0:
                    
                    return []
                goal = self.boxCentList[0]
                #print 'goalLoc before'
                goalLoc = self.findNearestGrid(goal)

                #truncate goalLoc.
                if goalLoc[0]>=len(self.Map) or goalLoc[1]>=len(self.Map[0]):
                    #find closest grid of goal
                    x = goalLoc[0]
                    y = goalLoc[1]
                    while x>=len(self.Map):
                        #print 'modify goal[0]'
                        x -= 1
                    while y>=len(self.Map[0]):
                        #print 'modify goal[1]'
                        y -= 1
                    
                    goalLoc=(x,y)
                    #print 'new goalLoc,',goalLoc
                self.valueMap = self.getValueMap(self.Map,goalLoc,goal)
                dist = robot.compute_distance((self.robot.x,self.robot.y),goal)
                #print 'get valueMap'
                step = self.getPathAndNextMove(startLoc,goal,goalLoc,dist)
                #print 'step',step
                return step
                
                
            elif self.holdBox == True:
                #go back to dropzone
                #print 'go back to dropzone'
                self.motion = (0.,0.)
                self.motionList=[]
                self.move = []                
                self.initRobot = (self.robot.x,self.robot.y,self.robot.bearing)
                #print 'initRobot Prop in return:',self.initRobot
                self.Prev_Path=[]
                
                startLoc = self.findNearestGrid((self.robot.x,self.robot.y))
                goal = (0.,0.) # init loc of robot
                goalLoc = self.findNearestGrid(goal)
                #print 'goalLoc in return',goalLoc
                self.valueMap = self.getValueMap(self.Map,goalLoc,goal)
                dist = robot.compute_distance((self.robot.x,self.robot.y),goal)
                step = self.getPathAndNextMove(startLoc,goal,goalLoc,dist)
                #print 'step',step
                return step
                
            #return []
    def StraightMoveToBox(self,goal):
        self.motion = (0.,0.)
        self.motionList=[]
        self.move = []
        dist = robot.compute_distance((self.robot.x,self.robot.y),goal)
        self.lastMove = True
        self.emptyPath =True
        self.lastMoveToDes(goal,True,0)
        move = ''
        motion = (0.,0.)
        if len(self.motionList)>0 and len(self.move)>0:
            motion = self.motionList[0]
            move = self.move[0]
        #else:
            #print 'move or motionList missing!!'

        #if move=='':
            #print 'move is WRONG!!!'
        return motion,move

        
    def getPathAndNextMove(self,startLoc,goal,goalLoc,dist):
        path = self.getFinalLocMap(startLoc,goalLoc)
        self.emptyPath = False
        self.lastMove = False
        if path ==[] or dist<1.0: # very close to last step                    
            self.lastMove = True
            self.emptyPath =True
            #print 'path is empty or dist is within 1.0!,should go last move'
            #print 'distance to boxCenter when path is empty:', robot.compute_distance((self.robot.x,self.robot.y), goal)
            self.lastMoveToDes(goal,True,0)
            #print 'self.Prev_Path',self.Prev_Path
            finalPath = self.Prev_Path
            #print 'finalPath',finalPath
            self.SetFinalMove(self.initRobot[0],self.initRobot[1],self.initRobot[2],finalPath)
            (self.robot.x,self.robot.y,self.robot.bearing)=self.initRobot
            if len(self.motionList)>0:
                self.motion = self.motionList[0]
                self.updateByMotioninMatrixs(self.motion[0],self.motion[1])
            return self.move[0]                    
        else: 
            self.SetPrevMove(path)
            #print 'path is not empty'
            self.lastMoveToDes(goal,True,0)
            finalPath = self.getFinalPath(self.Prev_Path)                
            self.SetFinalMove(self.initRobot[0],self.initRobot[1],self.initRobot[2],finalPath)

            (self.robot.x,self.robot.y,self.robot.bearing)=self.initRobot
            if len(self.move) !=0:                        
                if len(self.motionList)>0:
                    self.motion = self.motionList[0]
                self.updateByMotioninMatrixs(self.motion[0],self.motion[1])
                return self.move[0]
            
    def SetFinalMove(self,pos_x,pos_y,bearing,path):
        self.robot.x = pos_x
        self.robot.y = pos_y
        self.robot.bearing = bearing

        #move start from path[0]
        for i in range(1,len(path)):
            self.robotMovement(path[i],False)
            
    def getFinalPath(self,path):
        
        ind = 2 # check init + ind, if we can pass one point
        init = 0
        finalPath = []
        finalPath.append(path[0])

                                    

        while init+ind <len(path):
            start = path[init]
            end = path[init+ind]
            if self._is_traversable(start,end,robot.compute_distance(start,end)):
                ind += 1
            else:
                appendindex = init+ind-1
                finalPath.append(path[appendindex])
                init = init + ind -1
                ind =2

        if path[-1] not in finalPath:#append last step
            finalPath.append(path[-1])
        #print 'Tpath is:',path
        #print 'TfinalPath',finalPath
        return finalPath
    
    def lastMoveToDes(self,goal,isBox = True,num=None): #calculate the last move to destination
        while True:
            #print 'in while loop of lastMoveToDes'
            #print 'self.robot.x',self.robot.x,'self.robot.y',self.robot.y,'goal',goal
            distance,steering = self.robot.measure_distance_and_bearing_to(goal)
            if distance >=0.45: # not in the right range , needs to take another step
                #print 'distance',distance
                if self.emptyPath == False:
                
                    steering_now = min(self.robot.max_steering, max(-self.robot.max_steering, steering))
                    distance_now = min(self.robot.max_distance, max(0, distance))
                    bearing_now = robot.truncate_angle(self.robot.bearing + steering_now)
                    
                    if abs(steering) > self.robot.max_steering:
                        self.robot.bearing = bearing_now
                        
                        
                    else: # angle is ok
                        #print 'abs(steering) > self.robot.max_steering:'
                        distance_now -= 0.425

                        #check _is_traversable for current pos and next pos
                        nextPos_x = self.robot.x + (distance_now * math.cos(bearing_now))
                        nextPos_y = self.robot.y + (distance_now * math.sin(bearing_now))
                        
                        robotpos =(self.robot.x,self.robot.y)
                        robotNextPos = (nextPos_x,nextPos_y)
                        #print 'path traversalbe?:',self._is_traversable(robotpos,robotNextPos,distance_now)
                        if self._is_traversable(robotpos,robotNextPos,distance_now):
                            self.robot.x = robotNextPos[0]
                            self.robot.y = robotNextPos[1]
                            self.robot.bearing = bearing_now
                            self.Prev_Path.append((self.robot.x,self.robot.y))
                        
                        else:
                            if(isBox ==True):
                                #if it is not traversable:
                                #truncate goalLoc.
                                goalLoc = self.findNearestGrid(goal) 
                                if goalLoc[0]>=len(self.Map) or goalLoc[1]>=len(self.Map[0]):
                                    #find closest grid of goal
                                    x = goalLoc[0]
                                    y = goalLoc[1]
                                    while x>=len(self.Map):
                                        #print 'modify goal[0]'
                                        x -= 1
                                    while y>=len(self.Map[0]):
                                        #print 'modify goal[1]'
                                        y -= 1
                                    goalLoc=(x,y)
                                #gridLoc =                           
                                self.SetPrevMove([goalLoc])
                    break        
                elif self.emptyPath == True: #only go for the next step
                    distance = distance - 0.425
                    steering_now = min(self.robot.max_steering, max(-self.robot.max_steering, steering))
                    distance_now = min(self.robot.max_distance, max(0, distance))
                    bearing_now = robot.truncate_angle(self.robot.bearing + steering_now)

                    nextPos_x = self.robot.x + (distance_now * math.cos(bearing_now))
                    nextPos_y = self.robot.y + (distance_now * math.sin(bearing_now))
                    robotpos =(self.robot.x,self.robot.y)
                    robotNextPos = (nextPos_x,nextPos_y)
                    
                    
                    #change para of collision
                    self.ROBOT_RADIUS = 0.25
                    self.init_para_after_gettingMap()
                    self.ROBOT_RADIUS = 0.3
                    
                    #print 'robotNextPos,',robotNextPos,'robotpos',robotpos,'bearing_now',bearing_now,'distance_now',distance_now
                    

                    if self._is_traversable(robotpos,robotNextPos,distance_now):
                        #print 'it is traversable'
                        motion = (steering_now,distance)
                        self.motionList.append(motion)
                        move = 'move '+ str(steering_now) + ' ' + str(distance)
                        self.move.append(move)
                        break
                    else:
                        #print 'not traversable in the last move'
                        #
                        #truncate goalLoc.
                        goalLoc=self.findNearestGrid(goal)
                        if goalLoc[0]>=len(self.Map) or goalLoc[1]>=len(self.Map[0]):
                            #find closest grid of goal
                            x = goalLoc[0]
                            y = goalLoc[1]
                            while x>=len(self.Map):
                                #print 'modify goal[0]'
                                x -= 1
                            while y>=len(self.Map[0]):
                                #print 'modify goal[1]'
                                y -= 1
                            goalLoc=(x,y)
                        #gridLoc = 
                        self.SetPrevMove([goalLoc])
                        #print 'self.Prev_Path',self.Prev_Path
                        finalPath = self.Prev_Path
                        self.SetFinalMove(self.initRobot[0],self.initRobot[1],self.initRobot[2],finalPath)
                        
                        break
                        
                        
                    
            else:# dist in the range to get box
                # check if angle is ok.
                if(self.lastMove==True):
                    
                    #print 'lastMoveToDes: dist in the range to get box'
                    #print 'robot info:',(self.robot.x,self.robot.y,self.robot.bearing)
                    angle = math.atan2(goal[1]-self.robot.y,goal[0]-self.robot.x)
                    #print ' angle in the box for last move:',angle
                    #print 'abs(self.robot.bearing - angle)',abs(self.robot.bearing - angle)
                    if abs(robot.truncate_angle(self.robot.bearing - angle))>0.1:
                        delta_a = angle - self.robot.bearing

                        bearing = min(delta_a,self.max_steering)
                        bearing = max(delta_a,-self.max_steering)
                        bearing = robot.truncate_angle(bearing)

                        move = 'move '+ str(bearing) + ' ' + str(0.)
                        self.move.append(move)
                        self.motionList.append((bearing,0.))
                        break
                    else:
                        if self.holdBox == False: # lift box
                            #print 'angle is okay to get box'
                            move = 'lift '+str(0.)
                            self.move.append(move)
                            self.motionList.append((0.0,0.0))
                            self.holdBox =True  # set the flag to holdBox
                            #print 'move',move
                            self.boxDeliveredList.append(goal)
                            self.ActOnBox = True
                            break
                        else: # drop box down
                            #print 'angle is okay to drop box in the dropzone'
                            move = 'down '+str(0.)
                            #print 'move',move
                            self.move.append(move)
                            self.motionList.append((0.,0.0))
                            self.boxDelivered += 1
                            self.holdBox =False
                            self.ActOnBox= True
                            break
                else:
                    break
    
    def SetPrevMove(self,path):
        
        self.Prev_Path.append((self.robot.x,self.robot.y))
        #centerMap = self.getCenterMap(1)
        
        for p in path:
            center = self.gridCenterMap[p[0]][p[1]]
            self.robotMovement(center)

    def robotMovement(self,destination,gridMove=True):
        
        robotPos = (self.robot.x,self.robot.y)

        #print 'enter robotMovement,dist,',robot.compute_distance(robotPos,destination)
        while robot.compute_distance(robotPos,destination) > 1e-3:
            distance,steering = self.robot.measure_distance_and_bearing_to(destination)

            steering_now = max(-self.robot.max_steering, steering)
            if steering_now >self.robot.max_steering:
                steering_now = self.robot.max_steering
                
            distance_now = distance
            if self.robot.max_distance<distance:
                distance_now = self.robot.max_distance
                 

            #truncate angle
            angle = robot.truncate_angle(self.robot.bearing + steering_now)
            self.robot.bearing = angle

            if abs(steering) > self.robot.max_steering:#if the steering larger than what it can bear
                distance_now = 0#only turn but not bear

            #moving distance and coor calc
            self.robot.x += distance_now * math.cos(self.robot.bearing)
            self.robot.y += distance_now * math.sin(self.robot.bearing)

            robotPos = (self.robot.x,self.robot.y)#update robot position
            
            if gridMove==True:
                if distance_now > 1e-3:
                    self.Prev_Path.append((self.robot.x,self.robot.y))
            else:
                self.motionList.append((steering_now,distance_now))
                act = 'move '+str(steering_now)+' '+str(distance_now)
                self.move.append(act)

    def diffAngleMotionList(self,dist,lastMotion, diffAngle): #lastMotion = (0,1)
        
        if diffAngle > self.max_steering or diffAngle < -self.max_steering:
            if diffAngle >self.max_steering:
                n = int(diffAngle / self.max_steering)
                mod = diffAngle % self.max_steering
                m1 = (self.max_steering,0.)
                m2 = (mod,0.)
                m3 = lastMotion
                for i in range(n):
                    self.randomMotionList.append(m1)
                self.randomMotionList.append(m2)
                self.randomMotionList.append(m3)
            else:
                n =int(diffAngle / -self.max_steering)
                mod = diffAngle % -self.max_steering
                m1 = (-self.max_steering,0.)
                m2 = (mod,0.)
                m3 = (0.,1.)
                for i in range(n):
                    self.randomMotionList.append(m1)
                self.randomMotionList.append(m2)
                self.randomMotionList.append(m3)
            
        else:
            m1 = (diffAngle,0)
            m2 = lastMotion
            self.randomMotionList.append(m1)
            self.randomMotionList.append(m2)
    
    def random4DirWalk(self):#4 directions
        
        End = False
        start = (self.robot.x,self.robot.y)
        end = (0.,0.)
        motion = (0.001,0.001)
        dist = 1.0

        if len(self.randomMotionList)>0:
            #print self.randomMotionList
            motion = self.randomMotionList[0]
            self.randomMotionList.pop(0)
        else:          
            #print 'len(self.randomMotionList)==0'
            it =0
            while End == False and it <30:
                it += 1
                #up,down,left,right
                direction = ['up','down','left','right']
                index = random.randint(0,3)
                #print 'index',index
                # just turn
                if direction[index] =='up':
                    end = (start[0],start[1]+dist)
                    #check traversable
                    if self._is_traversable(start,end,robot.compute_distance(start,end)):
                        diffAngle = robot.truncate_angle(PI/2 - self.robot.bearing)
                        self.diffAngleMotionList(dist,(0.,1.), diffAngle)
                elif direction[index] =='down':
                    end = (start[0],start[1]-dist)
                    #check traversable
                    if self._is_traversable(start,end,robot.compute_distance(start,end)):
                        diffAngle = robot.truncate_angle(-PI/2 - self.robot.bearing)
                        self.diffAngleMotionList(dist,(0.,1.), diffAngle)
                        
                    
                elif direction[index] =='right':
                    #print 'before end'
                    #print 'start',start
                    end = (start[0]+dist,start[1])
                    #check traversable
                    #print 'before check traver'
                    if self._is_traversable(start,end,robot.compute_distance(start,end)):
                        diffAngle = robot.truncate_angle(0. - self.robot.bearing)
                        #print 'before diffAngleMotionList'
                        self.diffAngleMotionList(dist,(0.,1.), diffAngle)
                        
                        

                elif direction[index] =='left':                    
                    end = (start[0]-dist,start[1])
                    #check traversable
                    if self._is_traversable(start,end,robot.compute_distance(start,end)):
                        diffAngle = robot.truncate_angle(-PI - self.robot.bearing)
                        #print 'diffAngle',diffAngle                                              
                        self.diffAngleMotionList(dist,(0.,1.), diffAngle)
                        
                if len(self.randomMotionList)>0:
                    motion = self.randomMotionList[0]                        
                    steering,distance = motion[0],motion[1]          
                    
                    steering = max(-self.max_steering, steering)
                    steering = min(self.max_steering, steering)
                    distance = max(0, distance)
                    distance = min(self.max_distance, distance)
                         
                    #print 'iter',it
                    if self._is_traversable(start,end,robot.compute_distance(start,end)):
                        End = True
                        #print 'direction is:',direction[index]
                        break

            if it == 30:
                #print 'before randomWalk,set it to (0.,0.)'
                #motion = self.randomWalk()
                motion = (0.,0.)
                #print 'max 4 dir reached,iter:',it
            #after generating a list
            else:
                #print 'self.randomMotionList',self.randomMotionList
                motion = self.randomMotionList[0]
                self.randomMotionList.pop(0)    
            
        #startToP = (self.robot.x+4.5,self.robot.y-4.5)
        #endToP = (end[0]+4.5,end[1]-4.5)
        #print 'start',start,'end',end
        #print 'start',startToP,'end',endToP,'theta',theta,'dist',distance  
        return motion
    
        
    def randomWalkToCenter(self):#4 directions
        
        End = False
        start = (self.robot.x,self.robot.y)
        end = (0.,0.)
        motion = (0.001,0.001)
        dist = 1.0

        if len(self.randomMotionList)>0:
            print self.randomMotionList
            motion = self.randomMotionList[0]
            self.randomMotionList.pop(0)
        else:          
            #print 'len(self.randomMotionList)==0'
            it =0
            xlist = [round(start[0])-1.0,round(start[0]),round(start[0])+1.0]
            
            ylist = [round(start[1])-1.0,round(start[1]),round(start[1])+1.0]
            
            while End == False and it <100:
                it += 1
                ind_x = random.randint(0,2)
                ind_y = random.randint(0,2)
                #print 'before end'
                end = (xlist[ind_x],ylist[ind_y])
                
                dist = robot.compute_distance(start,end)
                #print 'after dist'
                if self._is_traversable(start,end,dist):
                    #print 'end',end
                    diffAngle = robot.truncate_angle(math.atan2(end[1]-self.robot.y,end[0]-self.robot.x))
                    #print 'after diffAngle'
                    self.diffAngleMotionList(dist,(0.,dist), diffAngle) #get randomMotionList
                    #print 'after'
                
                        
                if len(self.randomMotionList)>0:
                    #print 'enter randomMotionList'
                    motion = self.randomMotionList[0]                        
                    steering,distance = motion[0],motion[1]          
                    
                    steering = max(-self.max_steering, steering)
                    steering = min(self.max_steering, steering)
                    distance = max(0, distance)
                    distance = min(self.max_distance, distance)
                     
                    break

            if it == 100:
                #print 'before randomWalk,set it to (0.,0.)'
                #motion = self.randomWalk()
                motion = (0.,0.)
                #print 'max 100 iter reached,iter:',it
            #after generating a list
            else:
                #print 'self.randomMotionList',self.randomMotionList
                motion = self.randomMotionList[0]
                self.randomMotionList.pop(0)    
      
        #print 'start',startToP,'end',endToP,'theta',theta,'dist',distance  
        return motion
    
            
    def randomWalk(self):
        notEnd = True
        motion = (0.,0.)
        start = (self.robot.x,self.robot.y)
        end = (0.,0.)
        theta = 0.
        distance = 0.

        threeDir = False
        
        if threeDir ==False:   #random direction, random dist
            it = 0
            while(notEnd ==True and it<100):
                it += 1
                steering = random.uniform(-self.max_steering,self.max_steering)
                distance = min(random.random()*0.5,self.max_distance)
                
                steering = max(-self.max_steering, steering)
                steering = min(self.max_steering, steering)
                distance = max(0, distance)
                distance = min(self.max_distance, distance)
                
                motion = (steering,distance)
                
                
                theta = robot.truncate_angle(motion[0]+self.robot.bearing)
                
                
                end_x = motion[1] * math.cos(theta)+start[0]
                end_y = motion[1] * math.sin(theta)+start[1]
                end = (end_x,end_y)
                
                if self._is_traversable(start,end,robot.compute_distance(start,end)):
                    notEnd = False
            if it >=100:
                motion = (0.,0.)
                #print 'max iter in randomWalk()!!'
                
                #print 'self.NSWE',self.NSWE
        #startToP = (self.robot.x+4.5,self.robot.y-4.5)
        #endToP = (end[0]+4.5,end[1]-4.5)
        #print 'start',start,'end',end
        #print 'start',startToP,'end',endToP,'theta',theta,'dist',distance  
        return motion
    
    def checkIfCanMoveTo(self,motion):
        start = (self.robot.x,self.robot.y)
        distance = motion[1]
        steering = motion[0]


        steering = max(-self.max_steering, steering)
        steering = min(self.max_steering, steering)
        distance = max(0, distance)
        distance = min(self.max_distance, distance)        

        motion = (steering,distance)
        theta = robot.truncate_angle(motion[0]+self.robot.bearing)

        # add max, min constraints
        end_x = motion[1] * math.cos(theta)+start[0]
        end_y = motion[1] * math.sin(theta)+start[1]
        end = (end_x,end_y)
                
        if self._is_traversable(start,end,robot.compute_distance(start,end)):
            return True
        else:
            return False
    def getFinalLocMap(self,startLoc,endLoc):
        path = self.pathInValueMap(self.valueMap,startLoc,endLoc)
        if startLoc in path:
            path.remove(startLoc)
        if endLoc in path:
            path.remove(endLoc)
        
        return path
    
    def pathInValueMap(self,valueMap,startLoc,endLoc):
        startValue = valueMap[startLoc[0]][startLoc[1]]
        endValue = valueMap[endLoc[0]][endLoc[1]]
        path = []
        init = startLoc
        #print 'init',init
        notend = True
        it = 0
        path.append(startLoc)
        maxit = 20
        while(notend==True and it <=maxit):
            nearLocList = self.getLocNearGrid4Dir(self.Map,init)
            #print 'init is' + str(init)
            #print 'getnearLoc List' + str(nearLocList)
            it += 1
            if len(nearLocList)>0:
                valueList = []
                for nearLoc in nearLocList:
                    valueList.append(valueMap[nearLoc[0]][nearLoc[1]])
                #print 'valueList' + str(valueList)
                #print 'min index in valueList' + str(valueList.index(min(valueList)))
                init = tuple(nearLocList[valueList.index(min(valueList))]) ## might arise problem when two loc have the same value.
                if valueMap[init[0]][init[1]] != 0:
                    path.append(init)
                    if init in path:
                        #print 'repeated Path in valueMap'
                        break
            if valueMap[init[0]][init[1]] == 0 :
                notend = False
                path.append(init)
                if init in path:
                    #print 'repeated Path in valueMap'
                    break
            
        return path

    def printPathMap(self,path):
        pathMap = copy.deepcopy(self.Map)
        
        for i in path:
            pathMap[i[0]][i[1]]='*'

        print pathMap
                    
    
    def findNearestGrid(self,boxLoc):
        #boxLoc is coor
        nearGridLoc = (0.,0.)


        if self.EdgeNotEnough ==True:
            #print 'self.NSWE less than 2, in findNearestGrid'
            return []
        else:
            n = self.NSWE[0]
            w = self.NSWE[2]
            y = abs(int ((w-boxLoc[0]) / self.grid_width))  
            x = abs(int ((boxLoc[1]-n)/self.grid_height))
            
            nearGridLoc = (x,y)
        # row and column of grid array
        return nearGridLoc
    
    def buildGridMap(self):
        #1. associate segment points with box or wall
        #lmdict[key][0]: name, [1]:distance,[2]:relative bearing to robot bearing.[3]orderNum
        #print 'getSegList'
        boxSegList=self.getSegList('box')
        edgeSegList=self.getSegList('warehouse')
        wallSegList=self.getSegList('wall')

        #print 'edgeSegList',edgeSegList

        # sort wall from left to right
        #print 'wallSegList',wallSegList
        wallSegList.sort(key =itemgetter(0))
        #print 'wallSegList sorted',wallSegList

        

        self.edgeSegList=edgeSegList
        
        #print 'boxSegList',boxSegList
        #print 'edgeSegList',edgeSegList
        
        boxList = self.addToBoxList('box',boxSegList,0.7) # reset boxList, since coor might be changed a little bit.
        wallList = self.addToWallList('wall',wallSegList,1.3) # nearby wall will be collected as a big wall
        edgeList = self.addToEdgeList('edge',edgeSegList,1.5,0.2)#  and diff is less than 0.2

        
        
        currCoor = (self.robot.x,self.robot.y)
        #print 'boxList ',boxList
        #print 'wallList',wallList
        #print 'edgeList',edgeList
   
        #2. build map
        boxCentList = self.getCentListForMapping(currCoor,0.1,boxList,False)
        
        #print 'boxCentList before removing delivered',boxCentList


        removeList =[]
        for boxCent in boxCentList:
            for deliveredBox in self.boxDeliveredList:
                if robot.compute_distance(boxCent,deliveredBox) < 0.4:
                    #print 'dist between box center',robot.compute_distance(boxCent,deliveredBox)
                    removeList.append(boxCent)
                    

        if len(removeList)>0:
            for e in removeList:
                if e in boxCentList:
                    boxCentList.remove(e)
            

        
        #after remove delivered box
                    
        if len(boxCentList) >0 or self.holdBox ==True: #if holding a box, still consider one box showed
            self.boxShowed = True
        else:
            self.boxShowed = False

        #print 'boxCentList after removing delivered',boxCentList
        #print 'wallList',wallList   
        wallCentList = self.getCentListForMapping(currCoor,0.5,wallList,True)

        
        #for cent in wallCentList:
            
            #print 'wallCent',' '.join(format(f, '.3f') for f in cent)
            
        #print 'wallCentList:{:6.2f}'.format(wallCentList)        
        boarderList = []

        self.boxCentList  = boxCentList
        self.wallCentList = wallCentList
        self.edgeList = edgeList


        
        for edge in edgeList:
            #print 'edge is',edge
            if len(edge[1]) < 1:#No point at edge
                #print 'One edge not enough data'
                pass
            else:
                s = 0.
                s_string =''
                for i in range(len(edge[1])):
                    
                    if edge[0]=='N':                        
                        s_string ='y_limit ='
                        s += edge[1][i][1]
                        
                    elif edge[0]=='S':
                        s_string ='y_limit ='
                        s += edge[1][i][1]
                        
                    elif edge[0]=='W':
                        s_string ='x_limit ='
                        s += edge[1][i][0]
                        
                    elif edge[0]=='E':
                        s_string ='x_limit ='
                        s += edge[1][i][0]
                        
                    
                boarderList.append([s_string,s/float(len(edge[1]))])    
                                       
        #print 'boarderList after edge',boarderList

        if self.isBoxMap==False:
            self.Map = self.MappingFromDataLists(boxCentList,wallCentList,boarderList,self.gridNum)
            #print 'gridMap is:',self.Map
        else:#build box map
            if len(boxCentList)>0:
                boxToGet = boxCentList[0]
                self.Map = self.getBoxGridMap(boxCentList,wallCentList,self.gridNum)
            #else:
               # print 'no box showed in boxCentList'
        
        
    def getBoxGridMap(self,boxCentList,wallCentList,gridNum):
        pass

    def getValueMap(self,gridmap,goal,goalCent):
        valueMap = [[-100 for col in range(len(gridmap[0]))]for row in range(len(gridmap))]
        #print 'init valueMap',valueMap
        if len(self.Map)!=0:
            
            for i in range(len(gridmap)):
                for j in range(len(gridmap[0])):
                    if self.Map[i][j]=='#':
                        valueMap[i][j]=999
    
                           
            #print '#enter top grid just for test'

            #find closest grid of goal
            x = goal[0]
            y = goal[1]
            while goal[0]>=len(self.Map):
                #print 'modify goal[0]'
                x -= 1
            while goal[1]>=len(self.Map[0]):
                #print 'modify goal[1]'
                y -= 1
            
            goal=(x,y)
            #print 'new goal,',goal
            valueMap[goal[0]][goal[1]]=0
            
            NewPoint = True
            currLocList = []
            currLocList.append(goal)
            it=0
            while(NewPoint == True):
                it += 1
                if len(currLocList) >0:
                    currLoc = currLocList[0] #always get first element
                    nearLocList = []
                    nearLocList = self.getLocNearGrid4Dir(self.Map,currLoc)
                    
                    if len(nearLocList) > 0:
                        for nearLoc in nearLocList:
                            if valueMap[nearLoc[0]][nearLoc[1]]<0: # init -100 value, not assigned
                                valueMap[nearLoc[0]][nearLoc[1]] = valueMap[currLoc[0]][currLoc[1]]+1                            
                                currLocList.append(nearLoc)                       
                    currLocList.remove(currLoc)    
                    if len(currLocList)==0:
                        NewPoint = False


            
            for i in range(len(gridmap)):
                for j in range(len(gridmap[0])):
                    gridLoc = (i,j)
                    gridCent = self.gridCenterMap[gridLoc[0]][gridLoc[1]]
                    #check NSWE side ,block near grids.
                    #nearLocList =[]
                    nearLocList = self.getLocNearGrid4Dir(self.Map,goal)
                    nearLocList.append(goal)
                    
                    #print 'near goal loc list',nearLocList
                    if gridLoc not in nearLocList: # open the near loc of goal
                    
                        if gridCent[1]+self.grid_height/2.0>=self.NSWE[0]-self.ROBOT_RADIUS:
                            valueMap[i][j] += 0.5
                        if gridCent[1]-self.grid_height/2.0<=self.NSWE[1]+self.ROBOT_RADIUS:
                            valueMap[i][j] += 0.5
                        if gridCent[0]+self.grid_width/2.0>=self.NSWE[3]-self.ROBOT_RADIUS:
                            valueMap[i][j] += 0.5
                        if gridCent[0]-self.grid_width/2.0<=self.NSWE[2]+self.ROBOT_RADIUS:
                            valueMap[i][j] += 0.5
          
        
        return valueMap

    def getLocNearGrid4Dir(self,gridmap,loc):
        adjaList = []
        row=loc[0]
        column=loc[1]
        for i in range(-1,2):
            for j in range(-1,2):
                if row+i>=0 and row+i<len(gridmap) and column+j>=0 and column+j<len(gridmap[0]):
                    if (i+row!=row or j+column!=column):
                        if i ==0 or j==0:#4 directions
                            nearLoc =(row+i,column+j)
                            if gridmap[nearLoc[0]][nearLoc[1]] != '#':
                                # and  self.Map[nearLoc[0]][nearLoc[1]].isalnum() ==False:
                                
                               adjaList.append(nearLoc)
        return adjaList        
                       
        




        
    def MappingFromDataLists(self,boxCentList,wallCentList,boarderList,gridNum):
        self.gridNum = gridNum
        isCalced = False
        self.NSWE=[]

        #backup boarder
        if self.prev_boarder != [] and len(self.prev_boarder)> len(boarderList): # if we have back up, len of boarder shrinks
            boarderList = self.prev_boarder 
        
        if len(boarderList)<=2:
            self.EdgeNotEnough = True
            #print 'boarderList is too short:lessthan 2'
          
            self.prev_boarder = boarderList
            ymax,ymin,xmax,xmin = self.findMinCoor()
            self.NSWE = [ymax,ymin,xmin,xmax]
            
            
        elif len(boarderList)==3:
            # if boarderList ==3, make it up to a square
            #print 'boarderList is too short: len = 3'
            #self.EdgeNotEnough =True
            bound_NS=[]
            bound_WE=[]
            
            for bd in boarderList:
                if bd[0]=='x_limit =':
                    bound_WE.append(bd[1])
                elif bd[0]=='y_limit =':                    
                    bound_NS.append(bd[1])
            
            if len(bound_WE)==1: 
                sq_len = abs(bound_NS[0]-bound_NS[1])
                # determine bound_WE is west or east
                s = 0.
                for i in self.edgeSegList:
                    s += i[0]-bound_WE[0]
                if s>0:# it is west bound
                    bound_WE.append(bound_WE[0]+sq_len)
                else:
                    bound_WE.append(bound_WE[0]-sq_len)
            elif len(bound_NS)==1:
                sq_len = abs(bound_WE[0]-bound_WE[1])
                s = 0.
                #print 'before check ns edge'
                for i in self.edgeSegList:
                    s += i[1]-bound_NS[0]
                #print 'before check s of sum'
                if s>0:# it is south bound
                    bound_NS.append(bound_NS[0]+sq_len)
                else:
                    bound_NS.append(bound_NS[0]-sq_len)
            #print 'bound_WE',bound_WE,'bound_NS',bound_NS
            ####to be commented  
            
            self.NSWE = [max(bound_NS),min(bound_NS),min(bound_WE),max(bound_WE)]
           
            isCalced = True
            #backup boarder
            self.prev_boarder = boarderList
            
            boarderList.append('add extra edge by estimating squaring')
            #print 'boarderList len after append',len(boarderList)
            #return []
        if len(boarderList)==4:
            #backup boarder
            self.prev_boarder = boarderList
            
            self.EdgeNotEnough =False
            if isCalced == False:
                bound_NS=[]
                bound_WE=[]
                for bd in boarderList:
                    if bd[0]=='x_limit =':
                        bound_WE.append(bd[1])
                    elif bd[0]=='y_limit =':
                        
                        bound_NS.append(bd[1])

                self.NSWE = [max(bound_NS),min(bound_NS),min(bound_WE),max(bound_WE)]
                
            
            #ymax,ymin,xmax,xmin = self.findMinCoorInWallOrBox()
            ymax,ymin,xmax,xmin = self.findMinCoor()
            #print 'ymax,ymin,xmax,xmin',ymax,ymin,xmax,xmin
            self.NSWE = [max(ymax,self.NSWE[0]),min(ymin,self.NSWE[1]),min(xmin,self.NSWE[2]),max(xmax,self.NSWE[3])]
            
            
            alist = range(int(min(self.NSWE))-1,int(max(self.NSWE))+2)
            blist = []
            for a in range(len(alist)):
                alist[a] = float(alist[a] + 0.5)
                blist.append(alist[a])
            for j in range(len(self.NSWE)):
                diff = 1000.0
                index =1000
                for i in range(len(blist)):
                    diff_new = abs(blist[i]-self.NSWE[j])
                    if diff_new <diff:
                        diff = diff_new
                        index = i
                if index == 1000:
                    pass
                    #print 'round self.NSWE failed at line 1148'
                self.NSWE[j] = blist[index]
                
                    
            
                       

            #print 'boarderList after making up',boarderList
            #print 'self.NSWE',self.NSWE
            
            
            world_width = abs(self.NSWE[2]-self.NSWE[3])
            world_height = abs(self.NSWE[0]-self.NSWE[1])

            gridWidthNum = int(world_width)
            if world_width - int(world_width)>0.5:
                gridWidthNum += 1
             
            gridHeightNum = int(world_height)
            if world_height - int(world_height)>0.5:
                gridHeightNum += 1

            

            self.grid_width =world_width/ gridWidthNum
            self.grid_height=world_height/ gridHeightNum
            #print 'gridWidthNum',gridWidthNum,'gridHeightNum',gridHeightNum
            self.gridWidthNum=gridWidthNum
            self.gridHeightNum=gridHeightNum

            #print 'self.grid_width',self.grid_width,'self.grid_height',self.grid_height
            
            
            gridmap = [['.' for col in range(gridWidthNum)]for row in range(gridHeightNum)]
            #print 'original gridmap',gridmap
            self.gridCenterMap = self.getCenterMap(self.gridHeightNum,self.gridWidthNum)
            #print 'centermap',self.gridCenterMap
            #print 'after centermap'

            for b in boxCentList:
                for i in range(gridHeightNum):
                    for j in range(gridWidthNum):
                        gridLoc = (i,j)
                        if self.checkLandMarksInGrid(gridLoc,b,0.2)==True:
                            #if gridmap[i][j]!='.':
                                #print 'this grid was modified before!'
                            gridmap[i][j]='B'
            for w in wallCentList:
                for i in range(gridHeightNum):
                    for j in range(gridWidthNum):
                        gridLoc = (i,j)
                        if self.checkLandMarksInGrid(gridLoc,w,0.95)==True:
                            #if gridmap[i][j]!='.':
                                #print 'this grid was modified before!'
                            gridmap[i][j]='#'
            #print 'gridmap',gridmap
            return gridmap
            
        

    def getCenterMap(self,gridHeightNum,gridWidthNum):
        centerMap = [[(0,0) for col in range(gridWidthNum)]for row in range(gridHeightNum)]
        
        for i in range(gridHeightNum):
            #print 'i',i
            for j in range(gridWidthNum):
                centerMap[i][j] = self.getCoorGridCenter((i,j),self.gridNum)
        #print 'centerMap[2][3]',centerMap[2][3],'centerMap[1][3]',centerMap[1][3],'centerMap[0][3]',centerMap[0][3],
        return centerMap
    
    def getCoorGridCenter(self,gridLoc,gridNum):
        #gridLoc is the location in grid array/map ,(0,0) --> len(map)
        x = (gridLoc[1]+0.5)*self.grid_width + self.NSWE[2]
        y = -(gridLoc[0]+0.5)*self.grid_height + self.NSWE[0]
        center = (x,y)
        #print 'center',center
        #center = (self.NSWE[2]+gridLoc[1]*gridLength+gridLength/2.0,-(gridLoc[0]*gridLength+gridLength/2.0))
        return center    

    def checkLandMarksInGrid(self,gridLoc,landmarkCent,lmsize):
        gridCent = self.gridCenterMap[gridLoc[0]][gridLoc[1]]
        l1 = (gridCent[0]-self.grid_width/2,gridCent[1]+self.grid_height/2)
        r1 = (gridCent[0]+self.grid_width/2,gridCent[1]-self.grid_height/2)
        l2 = (landmarkCent[0] - lmsize/2,landmarkCent[1]+lmsize/2)
        r2 = (landmarkCent[0] + lmsize/2,landmarkCent[1]-lmsize/2)
        if self.checkRecIntersection(l1,r1,l2,r2)==True:
            return True
        else:
            return False
        
    def checkRecIntersection(self,l1,r1,l2,r2):
        
        if l1[0]>r2[0] or l2[0]>r1[0]:
            return False
        if l1[1]<r2[1] or l2[1]<r1[1]:
            return False
        return True
            
    def checkCloserToInt(self,segCoor):#return x:x is int, only change y
        a = int(segCoor[0])
        #intList =[a-1,a,a+1]

        min_dx =min(abs(segCoor[0]-a-1),abs(segCoor[0]-a),abs(segCoor[0]-a+1))

        b = int(segCoor[1])
        min_dy =min(abs(segCoor[1]-b-1),abs(segCoor[1]-b),abs(segCoor[1]-b+1))

        if min_dx <min_dy:
            return 'x'
        else:
            return 'y'
    def checkCloserToIntOrHalf(self,xcoor):
        intList = [int(xcoor)-1,int(xcoor),int(xcoor)+1]

        halfList = [float(int(xcoor))-1.5,float(int(xcoor))-0.5,float(int(xcoor))+0.5,float(int(xcoor))+1.5]

        intList = [abs(xcoor-float(intList[0])),abs(xcoor-float(intList[1])),abs(xcoor-float(intList[2]))]

        halfList = [abs(xcoor-float(halfList[0])),abs(xcoor-float(halfList[1])),abs(xcoor-float(halfList[2])),abs(xcoor-float(halfList[3]))]
    
            
        min_int = min(intList)
        min_half = min(halfList)


        if min_int<=min_half:
            return 'int'
        else:
            return 'half'
        
    def getWallCentListForLocalizing(self,currCoor,seglist,edgeList):
        errorL =[]
        for wall in seglist:
            #print 'wall',wall
            res_x = self.checkCloserToIntOrHalf(wall[0])
            res_y = self.checkCloserToIntOrHalf(wall[1])
            
            cent = (0.,0.)
            if (res_x =='int' and res_y=='int') or (res_x =='half' and res_y=='half'):
                pass
                #print 'int/half calc wrong: in localizing:',wall
                
            elif res_x =='int' and res_y=='half': #only need to change y
                if currCoor[1] > wall[1]:# robot is upon the point
                    cent = (wall[0],wall[1]-0.5)                                    
                else:#down side
                    cent = (wall[0],wall[1]+0.5)                    
            elif res_x =='half' and res_y=='int':
                if currCoor[0] > wall[0]:# robot is right of the point
                    cent = (wall[0]-0.5,wall[1])
                else:#left side
                    cent = (wall[0]+0.5,wall[1])# robot is left of the point 
            x = cent[0]
            y = cent[1]
            TrueCent = (round(x),round(y))
            #print 'TrueCent of wall:',TrueCent,'wall',wall

            error = (TrueCent[0]-cent[0],TrueCent[1]-cent[1])
            if cent ==(0.,0.):
                pass
                #print 'cent wasnt updated by localizing:wall'
            else:
                errorL.append(error)
        #print 'before edgelist'
        for wall in edgeList:
            cent = (wall[0],wall[1])
            x = cent[0]
            y = cent[1]
            
            poss_x = [int(x)+1.0,int(x)+0.5,int(x),int(x)-0.5,int(x)-1.0]
            #print 'poss_x',poss_x
            diff_xl =[abs(x-int(x)-1.0),abs(x-int(x)-0.5),abs(x-int(x)+0.),abs(x-int(x)+0.5),abs(x-int(x)+1.0)]
            #print 'diff_xl',diff_xl,'min',min(diff_xl)
            minX = 100.0
            index_x = 0
            for xl in diff_xl:
                if xl<minX:
                    minX = xl
                    index_x = diff_xl.index(xl)
            x_after = poss_x[index_x]


            poss_y = [int(y)+1.0,int(y)+0.5,int(y),int(y)-0.5,int(y)-1.0]
            diff_yl =[abs(y-int(y)-1.0),abs(y-int(y)-0.5),abs(y-int(y)+0.),abs(y-int(y)+0.5),abs(y-int(y)+1.0)]
            minY = 100.0
            index_y = 0
            for yl in diff_yl:
                if yl<minY:
                    minY = yl
                    index_y = diff_yl.index(yl)
            
            y_after = poss_y[index_y]
            
            TrueCent = (x_after,y_after)
            
            #print 'TrueCent of edge:',TrueCent,'wall',wall
            
            error = (TrueCent[0]-cent[0],TrueCent[1]-cent[1])
            #print 'edge error is:',error
            if cent ==(0.,0.):
                pass
                #print 'cent wasnt updated by localizing'
            else:
                errorL.append(error)

                
        #print 'before calc error'
        s = (0.,0.)
        s_x = 0.
        s_y = 0.
        for e in errorL:
            s_x += e[0]
            s_y += e[1]
        
        #print 'before diving by len(errorL):',len(errorL)
        if len(errorL)>0:
            s = (s_x/len(errorL),s_y/len(errorL))
        #else:
        #    print 'no element could be added to s:',s

        return s
            
                
        

        
    def getCentListForMapping(self,currCoor,size,boxList,isWall=False):
        #print 'boxList is empty or not:',boxList
        #this only for testing
        #boxList = [['wall0', [(-5.000000000247276, 0.5000000018166109), (-4.5000000004700675, 1.5938184105834807e-09)]], ['wall1', [(-4.499999999133315, 3.0000000016036092)]], ['wall2', [(-3.500000000024482, 1.0000000011482342), (-2.9999999999143077, 0.5000000007246841), (-2.4999999997404814, 1.0000000005508562)]], ['wall3', [(-3.4999999995788986, 2.0000000011482344), (-2.999999999361002, 2.5000000009205485)]], ['wall4', [(-3.4999999986877306, 4.0000000011482335), (-2.9999999989056274, 3.5000000009205485), (-2.4999999986926245, 4.000000000709994)]], ['wall5', [(-1.9999999995666546, 1.5000000003770295), (-1.4999999995788955, 2.0000000002693064)]], ['wall6', [(-1.9999999993610016, 2.5000000004651746)]], ['wall7', [(-0.5000000000048941, 0.9999999998359641), (8.569167597727301e-11, 0.4999999996817225)]], ['wall8', [(-0.49999999957399965, 1.9999999998286189)]], ['wall9', [(-0.4999999991333146, 2.9999999997821147), (1.0845833120498583e-09, 3.499999999618068)]]]
        #print 'wallList',boxList
        
        if boxList==[]:
            return []
        boxCentList=[]

        isWallList=[]
            
        
        for box in boxList:
            currCoorBefore = (currCoor[0],currCoor[1])
            if isWall == True:#change the currCoor to prev robotCoor
                #self.wallWithRobotCoor.append([coor,'robotCoor',robotcoor])#self.wallWithRobotCoor[i][0] to decide whcih i [2]:use robot pos
                for w in self.wallWithRobotCoor:
                    #if robot.compute_distance(w[0],box)<0.001:
                    #print 'w[0]',w[0],'box',box
                    for seg in box[1]:
                        if robot.compute_distance(w[0],seg)<0.1:
                            if currCoor != w[2]:
                                currCoor = w[2]
                                #print 'this function called, box',box
                        

            if currCoorBefore != currCoor:
                #print 'currCoor after',currCoor,'currCoor before',currCoorBefore
                pass
            #if box[1]==[(-0.49999999957399965, 1.9999999998286189)]:
                #print 'currCoor for tihs point is:',currCoor
            cent = (0.,0.)
            if len(box[1]) == 1:
                #set the center to the segment cent, not accurate -->for box
                if isWall ==False:
                    cent = box[1][0]
                    
                    x = cent[0]
                    y = cent[1]
                    
                    poss_x = [int(x)+1.0,int(x)+0.5,int(x),int(x)-0.5,int(x)-1.0]
                    #print 'poss_x',poss_x
                    diff_xl =[abs(x-int(x)-1.0),abs(x-int(x)-0.5),abs(x-int(x)+0.),abs(x-int(x)+0.5),abs(x-int(x)+1.0)]
                    #print 'diff_xl',diff_xl,'min',min(diff_xl)
                    minX = 100.0
                    index_x = 0
                    for xl in diff_xl:
                        if xl<minX:
                            minX = xl
                            index_x = diff_xl.index(xl)
                    x_after = poss_x[index_x]
                    poss_y = [int(y)+1.0,int(y)+0.5,int(y),int(y)-0.5,int(y)-1.0]
                    diff_yl =[abs(y-int(y)-1.0),abs(y-int(y)-0.5),abs(y-int(y)+0.),abs(y-int(y)+0.5),abs(y-int(y)+1.0)]
                    minY = 100.0
                    index_y = 0
                    for yl in diff_yl:
                        if yl<minY:
                            minY = yl
                            index_y = diff_yl.index(yl)
                    
                    y_after = poss_y[index_y]

                    cent = (x_after,y_after)
                    
                    boxCentList.append(cent)
                else: # for wall, not working
                    #box[1] = [(-4.499999999133315, 3.0000000016036092)]
                    #print 'box[1]',box[1]
                    #box[1]
                    res_x = self.checkCloserToIntOrHalf(box[1][0][0])
                    res_y = self.checkCloserToIntOrHalf(box[1][0][1])

                    if res_x =='int' and res_y=='int':
                        pass
                        #print 'int/half calc wrong: box[1]:',box[1]
                    elif res_x =='half' and res_y=='half':
                        pass
                        #print 'int/half calc wrong: box[1]:',box[1]
                    elif res_x =='int' and res_y=='half': #only need to change y
                        if currCoor[1] > box[1][0][1]:# robot is upon the point
                            cent = (box[1][0][0],box[1][0][1]-0.5)
                        else:#down side
                            cent = (box[1][0][0],box[1][0][1]+0.5)       
                    elif res_x =='half' and res_y=='int':
                        if currCoor[0] > box[1][0][0]:# robot is right of the point
                            cent = (box[1][0][0]-0.5,box[1][0][1])    
                        else:#left side
                            cent = (box[1][0][0]+0.5,box[1][0][1])# robot is left of the point
                            
                    cent = (round(cent[0]),round(cent[1]))
                    if cent != (0.,0.):
                        boxCentList.append(cent)    



            else:
                if len(box[1])==2: # has to be adjacent for box but not for wall
                    p1 = box[1][0]
                    p2 = box[1][1]

                    if isWall == True:                       

                        res_x1 = self.checkCloserToIntOrHalf(p1[0])
                        res_y1= self.checkCloserToIntOrHalf(p1[1])
                        res_x2=self.checkCloserToIntOrHalf(p2[0])
                        res_y2=self.checkCloserToIntOrHalf(p2[1])

                        if res_x1 == res_y1 or res_x2==res_y2 or p1==p2:
                            pass
                            #print 'calc int/half wrong, two points,p1,',p1,'p2',p2

                        if res_x1 == 'int' and res_y1 =='half':
                            if res_x2 == 'int' and res_y2 =='half':#parallel
                                cent =((p1[0]+p2[0])/2.0,(p1[1]+p2[1])/2.0)
                            elif  res_x2 == 'half' and res_y2 =='int':#near two
                                cent = (p1[0],p2[1])
                        elif res_x1 == 'half' and res_y1 =='int':
                            if res_x2 == 'int' and res_y2 =='half':#near two
                                cent =(p2[0],p1[1])
                            elif  res_x2 == 'half' and res_y2 =='int':#parallel
                                cent = ((p1[0]+p2[0])/2.0,(p1[1]+p2[1])/2.0)
                            
                            
                        

                    
                    
                    
                    
                    elif isWall ==False:
                        parallel = False
                        if min(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))<size/10.0:
                            #it is parallel
                            parallel = True
                        if parallel == True:
                            cent = ((p1[0]+p2[0])/2.0,(p1[1]+p2[1])/2.0)
                            #print 'two points are parallel'
                        else:    
                            if currCoor[0]<min(p1[0],p2[0]):
                                
                                if currCoor[1]<min(p1[1],p2[1]):#bottom_left robot
                                    cent = (max(p1[0],p2[0]),max(p1[1],p2[1]))
                                    
                                elif currCoor[1]>max(p1[1],p2[1]): #up_left
                                    cent = (max(p1[0],p2[0]),min(p1[1],p2[1]))
                                else: #y in two points
                                    if p1[0]<p2[0]:
                                        cent = (p1[0]+size,p1[1])
                                    else:
                                        cent = (p2[0]+size,p2[1])                                              
                                    #print 'robot is parallel to box'
                            elif currCoor[0]>max(p1[0],p2[0]):
                                if currCoor[1]<min(p1[1],p2[1]):#bottom_Right
                                    cent = (min(p1[0],p2[0]),max(p1[1],p2[1]))
                                elif currCoor[1]>max(p1[1],p2[1]): #up_right
                                    cent = (min(p1[0],p2[0]),min(p1[1],p2[1]))
                                else:#y in two points
                                    if p1[0]>p2[0]:
                                        cent = (p1[0]-size,p1[1])
                                    else:
                                        cent = (p2[0]-size,p2[1])                                              
                                    #print 'robot is parallel to box'
                            else: # x in two points
                                if currCoor[1]<min(p1[1],p2[1]):# on bot
                                    if p1[1]<p2[1]:
                                        cent = (p1[0],p1[1]+size)
                                    else:
                                        cent = (p2[0],p2[1]+size)
                                else:
                                    if p1[1]<p2[1]:
                                        cent = (p1[0],p1[1]-size)
                                    else:
                                        cent = (p2[0],p2[1]-size)
                                #print 'robot is parallel to box'
                    
                                                                     
                elif len(box[1])>2 and len(box[1])<=4:#more than 3 points
                    p1 = box[1][0]
                    p2 = box[1][1]
                    p3 = box[1][2]
                    dx1 = abs(p1[0]-p2[0])
                    dx2 = abs(p2[0]-p3[0])
                    dx3 = abs(p3[0]-p1[0])                
                    dy1 = abs(p1[1]-p2[1])
                    dy2 = abs(p2[1]-p3[1])
                    dy3 = abs(p3[1]-p1[1])

                    if min(dx1,dy1) == min(min(dx1,dy1),min(dx2,dy2),min(dx3,dy3)):
                        cent = (p1[0]+p2[0],p1[1]+p2[1])
                        cent = (cent[0]/2,cent[1]/2)
                    elif min(dx2,dy2) == min(min(dx1,dy1),min(dx2,dy2),min(dx3,dy3)):
                        cent = (p3[0]+p2[0],p3[1]+p2[1])
                        cent = (cent[0]/2,cent[1]/2)
                    elif min(dx3,dy3) == min(min(dx1,dy1),min(dx2,dy2),min(dx3,dy3)):
                        cent = (p3[0]+p1[0],p3[1]+p1[1])
                        cent = (cent[0]/2,cent[1]/2)
                elif len(box[1])>4:
                    pass
                    #print 'Segment of one Landmark bigger than 4!!!!!!!'
                if cent ==(0.,0.):
                    pass
                    #print 'cent isnt updated from(0.,0.)'
                    
                if isWall == True:
                    cent = (round(cent[0]),round(cent[1]))
                else:
                    x = cent[0]
                    y = cent[1]
                    
                    poss_x = [int(x)+1.0,int(x)+0.5,int(x),int(x)-0.5,int(x)-1.0]
                    #print 'poss_x',poss_x
                    diff_xl =[abs(x-int(x)-1.0),abs(x-int(x)-0.5),abs(x-int(x)+0.),abs(x-int(x)+0.5),abs(x-int(x)+1.0)]
                    #print 'diff_xl',diff_xl,'min',min(diff_xl)
                    minX = 100.0
                    index_x = 0
                    for xl in diff_xl:
                        if xl<minX:
                            minX = xl
                            index_x = diff_xl.index(xl)
                    x_after = poss_x[index_x]
                    poss_y = [int(y)+1.0,int(y)+0.5,int(y),int(y)-0.5,int(y)-1.0]
                    diff_yl =[abs(y-int(y)-1.0),abs(y-int(y)-0.5),abs(y-int(y)+0.),abs(y-int(y)+0.5),abs(y-int(y)+1.0)]
                    minY = 100.0
                    index_y = 0
                    for yl in diff_yl:
                        if yl<minY:
                            minY = yl
                            index_y = diff_yl.index(yl)
                    
                    y_after = poss_y[index_y]

                    cent = (x_after,y_after)


            
                if cent!=(0.,0.):
                    boxCentList.append(cent)
        
        #check list
        Duplicate =False
        centerRemoveList=[]
        newCenterList =[]
        c = 0
        for center in boxCentList:
            for another in boxCentList:
                if center !=another and robot.compute_distance(center,another) <0.3:
                    center_new = ((center[0]+another[0])/2.0,(center[1]+another[1])/2.0)
                    #print 'two centers too close!!!!Remove old two, center,',center,'another',another
                    #print 'merged new Center,',center_new
                    newCenterList.append(center_new)
                    centerRemoveList.append(center)
                    centerRemoveList.append(another)
                    c += 1

        for cent in centerRemoveList:
            if cent in boxCentList:
                boxCentList.remove(cent)
        for cent in newCenterList:
            boxCentList.append(cent)
        if c>0:
            pass
            #print 'removed duplicate center num:',c
                    
                    
        return boxCentList
        
    def getSegList(self,name):
        # avg coor of LM: (self.LMCoorDict[lmID][5][0][0]/self.LMCoorDict[lmID][5][1],self.LMCoorDict[lmID][5][0][1]/self.LMCoorDict[lmID][5][1])
        
        #print ' enter getSegLIst'
        boxSegList=[]
        # key:ID/ 'robot' [0]:name 1:index 2:coor 3:corresponding robot Pos 4: list of coor/ ???/robot pos, 5: (sumx,sumy),sum weight
        #self.LMCoorDict[lmID]=[self.LMDict[lmID][0],index,(dx,dy),(robotx,roboty)]
        for key in self.AllLMIDList:
            if self.LMCoorDict[key][0]==name:
                #print 'name == name'
                #print self.LMCoorDict[key]
                coor = (self.LMCoorDict[key][2][0] +self.LMCoorDict[key][3][0],self.LMCoorDict[key][2][1]+self.LMCoorDict[key][3][1])
                boxSegList.append(coor)
        if name =='wall':
            self.wallWithRobotCoor = []
            for key in self.AllLMIDList:
                if self.LMCoorDict[key][0]==name:
                    coor = (self.LMCoorDict[key][2][0] +self.LMCoorDict[key][3][0],self.LMCoorDict[key][2][1]+self.LMCoorDict[key][3][1])
                    robotcoor = self.LMCoorDict[key][3]
                    self.wallWithRobotCoor.append([coor,'robotCoor',robotcoor])#self.wallWithRobotCoor[i][0] to decide whcih i [2]:use robot pos
            #print 'self.wallWithRobotCoor',self.wallWithRobotCoor
        
        return boxSegList
    def findMinCoor(self): #
        #print ' enter getSegLIst'
        xList=[]
        yList=[]
        # key:ID/ 'robot' [0]:name 1:index 2:coor 3:corresponding robot Pos
        #self.LMCoorDict[lmID]=[self.LMDict[lmID][0],index,(dx,dy),(robotx,roboty)]
        for key in self.AllLMIDList:
            if self.LMCoorDict[key][0]!='robot':
                #print 'name == name'
                #print self.LMCoorDict[key]
                xcoor = self.LMCoorDict[key][2][0] +self.LMCoorDict[key][3][0]
                ycoor = self.LMCoorDict[key][2][1]+self.LMCoorDict[key][3][1]
                xList.append(xcoor)
                yList.append(ycoor)
        
        return max(yList),min(yList),max(xList),min(xList)

    def findMinCoorInWallOrBox(self):# finMinCoor in wall or 
        #print ' enter getSegLIst'
        xList=[]
        yList=[]
        # key:ID/ 'robot' [0]:name 1:index 2:coor 3:corresponding robot Pos
        #self.LMCoorDict[lmID]=[self.LMDict[lmID][0],index,(dx,dy),(robotx,roboty)]
        for key in self.AllLMIDList:
            if self.LMCoorDict[key][0]=='wall' or self.LMCoorDict[key][0]=='box' :
                #print 'name == name'
                #print self.LMCoorDict[key]
                xcoor = self.LMCoorDict[key][2][0] +self.LMCoorDict[key][3][0]
                ycoor = self.LMCoorDict[key][2][1]+self.LMCoorDict[key][3][1]
                xList.append(xcoor)
                yList.append(ycoor)
        
        return max(yList),min(yList),max(xList),min(xList)
                    
    def addToBoxList(self,name,segList,distThreshold):
        if segList == []:
            return []
        boxList = []
        if len(boxList)==0:
            key = name+str(len(boxList))
            #boxList[0]: name [1]:segmentCenterList
            boxList.append([key,[]])
            #print 'boxList[0]',boxList[0]
        for seg in segList:
            recorded = False
            sameBox = False
            
            for box in boxList:
                bSeg = box[1]
                #print 'box[1]',box[1]
                if seg in bSeg:
                    recorded = True
                    break
                for boxSeg in bSeg:
                    # check for all boxSeg
                    if robot.compute_distance(boxSeg,seg) < distThreshold:
                        all_good =True
                        for b in bSeg:
                            if not robot.compute_distance(b,seg) < distThreshold:
                                all_good =False
                        if all_good ==True:
                            sameBox = True
                            bSeg.append(seg)                    
                        break
                        #print 'sameBox is true'
                        
            if recorded == False and sameBox ==False:
                for box in boxList: # need a new box for this new seg
                    if box[1]==[]:#this box is empty
                        box[1].append(seg)
                        break
                    else:
                        key = name+str(len(boxList))
                        boxList.append([key,[]])
                        boxList[-1][1].append(seg)
                        break
        return boxList
        
    def addToWallList(self,name,segList,distThreshold):
        if segList == []:
            return []
        wallList = []
        if len(wallList)==0:
            key = name+str(len(wallList))
            #boxList[0]: name [1]:segmentCenterList
            wallList.append([key,[]])
        #print 'wallSegList',segList
        for seg in segList:
            sameBox = False
            
            for box in wallList:
                if sameBox ==True:
                    break
                bSeg = box[1]
                #print 'box[1]',box[1]
                
                for boxSeg in bSeg:
                    if sameBox == True:
                        break
                
                    # check for all boxSeg
                    xdiff = abs(boxSeg[0]-seg[0])
                    ydiff = abs(boxSeg[1]-seg[1])
                    if robot.compute_distance(boxSeg,seg) < distThreshold and min(xdiff,ydiff)>0.2: # not parallel wall
                        all_good =True
                        for b in bSeg: # for all segments in the same wall, all meet such condition
                            if robot.compute_distance(b,seg) > distThreshold:
                                all_good =False
                        if all_good ==True:
                            if sameBox ==False:
                                sameBox = True
                                bSeg.append(seg)                    
                                break
                        #print 'sameBox is true'
                    
            if sameBox ==False: #re-check the box, but change the condition
                if True:
                    for box in wallList: # need a new box for this new seg
                        if box[1]==[]:#this box is empty
                            box[1].append(seg)
                            break
                        else:
                            key = name+str(len(wallList))
                            wallList.append([key,[]])
                            wallList[-1][1].append(seg)
                            break
     
                
                
        
        return wallList

        
    def addToEdgeList(self,name,segList,distThreshold,mdiffTol):
        edgeList=[]
        
        edgeList.append(['N',[]])
        edgeList.append(['S',[]])
        edgeList.append(['W',[]])
        edgeList.append(['E',[]])

        y_max = segList[0][1]
        y_min = segList[0][1]
        x_max = segList[0][0]
        x_min = segList[0][0]
        #print y_max,y_min,x_max,x_min
        for edgeSeg in segList:
            if edgeSeg[0]>x_max:
                x_max = edgeSeg[0]
            if edgeSeg[0]<x_min:
                x_min = edgeSeg[0] 
            if edgeSeg[1]<y_min:
                y_min = edgeSeg[1]            
            if edgeSeg[1]>y_max:
                y_max = edgeSeg[1]
        leftoverList=[]
        #print 'before 2nd loop'
        for edgeSeg in segList:
            #print 'edgeSeg',edgeSeg
            #print 'y_max',y_max
            if abs(edgeSeg[1] - y_max) <mdiffTol:
                #print 'after if state'
                edgeList[0][1].append(edgeSeg)
            elif abs(edgeSeg[1] - y_min)<mdiffTol:
                edgeList[1][1].append(edgeSeg)
            elif abs(edgeSeg[0] - x_min)<mdiffTol:
                edgeList[2][1].append(edgeSeg)
            elif abs(edgeSeg[0] - x_max)<mdiffTol:
                edgeList[3][1].append(edgeSeg)
            else:
                leftoverList.append(edgeSeg)

        if len(leftoverList)>0:
            pass
        return edgeList
             
    def set_robot_state(self, robot_has_box, robot_is_crashed, boxes_delivered, verbose = False):        
        
        self.robot_has_box = robot_has_box
        self.robot_is_crashed = robot_is_crashed
        self.boxes_delivered = boxes_delivered

        if verbose:
            print "Robot has box: {},  Robot is crashed: {}".format(robot_has_box, robot_is_crashed)
            print "Boxes delivered: {}".format(boxes_delivered)



    


    
    
