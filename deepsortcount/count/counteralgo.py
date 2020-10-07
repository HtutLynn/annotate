import numpy as np

class Counter:
    """
    Implementation of the Bboxes path algorithm
    If a tracked object's midpoint pass through the designated bboxes related to a certain door/entry
    and then dissappears in a couple of frames later, then
    that object is counted as have entered the into the entry/door

    Parameters
    ----------
    doors : a List
        Contain DoorInfo class instances
    stats : an Object
        A class used to store people in doors

    Attributes
    ----------
    """

    def __init__(self, doors):
        """
        Initiate the BBoxes path algorithm
        
        """
        self.doors = doors
        self.doors_count = len(doors)
        self.max_count_age = 20

    @staticmethod
    def _check_if_object_in_box(tracked_object, door):
        """
        Check if a centroid/point of a object detection rectangle 
        is in checkpoint box

        Parameters
        ----------
        tracked_object: An instance of the `track` class
            An `track` instance containing infos such as centroids, measurements
        door : An instance of the `DoorInfo` class
            An `DoorInfo` class instance cntaining info such as checkpoints

        Returns
        -------
        return_list : A List containing two Boolean values    
        """

        tracked_object = tracked_object
        door = door
        return_list = []
        x_mid, y_mid = tracked_object.centroid
        x0_min, y0_min, x0_max, y0_max = door.first_checkpoint_box
        x1_min, y1_min, x1_max, y1_max = door.second_checkpoint_box

        if x_mid < x0_max and x_mid > x0_min:
            if y_mid < y0_max and y_mid > y0_min:
                return_list.append(True)
            else:
                return_list.append(False)
        else:
            return_list.append(False)   

        if x_mid < x1_max and x_mid > x1_min:
            if y_mid < y1_max and y_mid > y1_min:
                return_list.append(True)
            else:
                return_list.append(False)
        else: 
            return_list.append(False)

        return return_list

    def count(self, tracks, stats):
        """
        Count the number of people went in/out of doors

        Parameters
        ----------
        tracks : a List
            A list containing all current `track` instances in tracker

        
        Returns
        -------
        curated_tracks : List
            A list containing curated tracks
        stats : An instance of `stats` class
            A class which have the number of people in each door
        """
        curated_tracks_ids = []

        # Do counting process on every track
        for track_idx, track in enumerate(tracks):

            # Process on newly detected track but hasn't assigned a track id yet
            if track.is_tentative():
                """
                    There is only one route in counting algorithm if the track's state is tentative
                    1. Tentative state means that the object is still very immature and the deepsort tracking algorithm
                    is not sure whether to track this one or not. Therefore, we will only take the bool_list of 
                    the track's first detected time step to count if they tracked object has exited from the one of the rooms or not
                """
                if track.age == 1:
                    # check for every door
                    for no, door in enumerate(self.doors):
                        # calculate whether the object is in door checkpoints
                        bool_list = self._check_if_object_in_box(track, door)
                        track.checkpoints[no]['first_checkpoint'] = bool_list[0]
                        track.checkpoints[no]['second_checkpoint'] = bool_list[1]
                
                curated_tracks_ids.append(track_idx)

            # Process on already confirmed tracked objects
            elif track.is_confirmed():
                """
                There are three main routes in counting algorithm if the current tracked object is confirmed
                1. If the tracked object's age is equal to 3, which means it is freshly tracked object,
                therefore, it is decided that this is a recurring object which is qualified to be tracked.
                Therefore it is required to check whether it has exited from one of the doors.

                2. If the tracked object's age is greater than 3 and it's time_since_update count is less than count_age
                then we can assume that the tracked object is still roaming in scence and
                therefore find out if that object has hit through the checkpoints of all doors or not
                
                3. If the tracked object is not detected for max_count_age (different from max_age of tracking),
                Check if that tracked object's centroid was in a door's second checkpoint box and at the same time,
                if the tracked object's current door first checkpoint was also flagged as True, then we can safely assume that
                the object has went into the room. If above condition is not satisfied, then we can set that respective door's
                checkkpoints to False and then leave it alone to see if the tracker will pick it again or not
                """
                
                if track.age == 3:
                    for no, door in enumerate(self.doors):
                        if track.checkpoints[no]['second_checkpoint'] == True:
                            stats.Doors[no] = stats.Doors[no] - 1
                            track.checkpoints[no]['second_checkpoint'] = False
                            bool_list = self._check_if_object_in_box(track, door)
                            track.checkpoints[no]['first_checkpoint'] = bool_list[0]
                            track.checkpoints[no]['second_checkpoint'] = bool_list[1]
                        else:
                            bool_list = self._check_if_object_in_box(track, door)
                            if bool_list[0]:
                                track.checkpoints[no]['first_checkpoint'] = bool_list[0]
                            if bool_list[1]:
                                track.checkpoints[no]['second_checkpoint'] = bool_list[1]
                    
                    curated_tracks_ids.append(track_idx)

                elif track.age > 3 and track.time_since_update < self.max_count_age:
                    for no, door in enumerate(self.doors):
                        bool_list = self._check_if_object_in_box(track, door)

                        if bool_list[0]:
                            track.checkpoints[no]['first_checkpoint'] = bool_list[0]
                        if bool_list[1]:
                            track.checkpoints[no]['second_checkpoint'] = bool_list[1]
                    
                    curated_tracks_ids.append(track_idx)

                elif track.time_since_update >= self.max_count_age:
                    delete_flag = False
                    for no, door in enumerate(self.doors):
                        bool_list = self._check_if_object_in_box(track, door)

                        if bool_list[1] and track.checkpoints[no]['first_checkpoint']:
                            stats.Doors[no] = stats.Doors[no] + 1
                            delete_flag = True
                        else:
                            # further modifications can be made here.
                            # requires better solution than this
                            track.checkpoints[no]['first_checkpoint'] = False
                            track.checkpoints[no]['second_checkpoint'] = False

                    if not delete_flag:
                        curated_tracks_ids.append(track_idx)
                        
            # process on the  deleted tracks
            elif track.is_deleted():

                # if the track is already set to deleted mode by deepsort track,
                # then let the tracker handles it what to do with it
                curated_tracks_ids.append(track_idx)

        curated_tracks = [tracks[idx] for idx in curated_tracks_ids]
        
        return curated_tracks, stats
