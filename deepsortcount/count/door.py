# Information class about a door 

class DoorInfo:
    def __init__(self, first_checkpoint_box, second_checkpoint_box):
        """
        Checkpoint bounding boxes setter for each door 
        """
        self.first_checkpoint_box = first_checkpoint_box
        self.second_checkpoint_box = second_checkpoint_box