import os
import time
import numpy as np
import argparse
import threading
import screen.record_screen as screen_recorder
import cv2
import pygame
from PIL import Image
from keyboard.getkeys import key_check


pygame.init()

cartella = "C:\\Users\\39347\\Desktop\\Self-Driving-Car-in-Video-Games-master\\output_directory"


def counter_coordinates(output: np.ndarray) -> int:
    if np.allclose(output, [-0.75,0.75], atol=0.25):
        print("1")
        return 1
    elif np.allclose(output, [-0.25,0.75], atol=0.25):
        print("2")
        return 2
    elif np.allclose(output, [0.25,0.75], atol=0.25):
        print("3")
        return 3
    elif np.allclose(output, [0.75,0.75], atol=0.25):
        print("4")
        return 4
    elif np.allclose(output, [-0.75,0.25], atol=0.25):
        print("5")
        return 5
    elif np.allclose(output, [-0.25,0.25], atol=0.25):
        print("6")
        return 6
    elif np.allclose(output, [0.25,0.25], atol=0.25):
        print("7")
        return 7
    elif np.allclose(output, [0.75,0.25], atol=0.25):
        print("8")
        return 8
    elif np.allclose(output, [-0.75,-0.25], atol=0.25):
        print("9")
        return 9
    elif np.allclose(output, [-0.25,-0.25], atol=0.25):
        print("10")
        return 10
    elif np.allclose(output, [0.25,-0.25], atol=0.25):
        print("11")
        return 11
    elif np.allclose(output, [0.75,-0.25], atol=0.25):
        print("12")
        return 12
    elif np.allclose(output, [-0.75,-0.75], atol=0.25):
        print("13")
        return 13
    elif np.allclose(output, [-0.25,-0.75], atol=0.25):
        print("14")
        return 14
    elif np.allclose(output, [0.25, -0.75], atol=0.25):
        print("15")
        return 15
    elif np.allclose(output, [0.75, -0.75], atol=0.25):
        print("16")
        return 16

    
def save_data(dir_path: str, images: np.ndarray, y: int, number: int):
    """
    Save a trainign example
    Input:
     - dir_path path of the directory where the files are going to be stored
     - data numpy ndarray
     - number integer used to name the file
    Output:

    """
    lista = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    if y in lista:
        y = f"0{y}"
        Image.fromarray(
            cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
        ).save(os.path.join(dir_path, f"{number}_{y}.jpeg"))
    else:
        Image.fromarray(
            cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
        ).save(os.path.join(dir_path, f"{number}_{y}.jpeg"))


def get_last_file_num(dir_path: str) -> int:
    """
    Given a directory with files in the format [number].jpeg return the higher number
    Input:
     - dir_path path of the directory where the files are stored
    Output:
     - int max number in the directory. -1 if no file exits
     """

    files = [
        int(f.split(".")[0])
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jpeg")
    ]

    return -1 if len(files) == 0 else max(files)

# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

textPrint = TextPrint()

def generate_dataset(output_dir: str, use_probability: bool = True) -> None:
    """
    Generate dataset exampled from a human playing a videogame
    HOWTO:
        Set your game in windowed mode
        Set your game to 1600x900 resolution
        Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
        Play the game! The program will capture your screen and generate the training examples. There will be saved
         as files named "training_dataX.npz" (numpy compressed array). Don't worry if you re-launch this script,
          the program will search for already existing dataset files in the directory and it won't overwrite them.

    Input:
    - output_dir: Directory where the training files will be saved
    - num_training_examples_per_file: Number of training examples per output file
    - use_probability: Use probability to generate a balanced dataset. Each example will have a probability that
      depends on the number of instances with the same key combination in the dataset.

    Output:

    """
    if os.path.exists(output_dir):
        print(f"{output_dir} exists!. I will use the existent one.")
    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exits. We will create it.")
        os.makedirs(output_dir)

    training_data: list = []
    stop_recording: threading.Event = threading.Event()

    th_img: threading.Thread = threading.Thread(
        target=screen_recorder.img_thread, args=[stop_recording]
    )

    th_seq: threading.Thread = threading.Thread(
        target=screen_recorder.image_sequencer_thread, args=[stop_recording]
    )
    th_img.setDaemon(True)
    th_seq.setDaemon(True)
    th_img.start()
    # Wait to launch the image_sequencer_thread, it needs the img_thread to be running
    time.sleep(2)
    th_seq.start()
    number_of_files: int = get_last_file_num(output_dir) + 1
    time.sleep(6)
    last_num: int = 5  # The image sequence starts with images containing zeros, wait until it is filled
    
    training_data = []
    
    while True:

        paused = False

        while last_num == screen_recorder.num:
            time.sleep(0.01)

        last_num = screen_recorder.num
        img_seq, output = screen_recorder.seq.copy(), screen_recorder.key_out.copy()
    
        print(
            f"Recording at {screen_recorder.fps} FPS\n"
            f"Images in sequence {len(img_seq)}\n"
            f"Training data len {number_of_files} sequences\n"
            f"Number of archives {number_of_files}\n"
            f"Push QE to exit\n",
            end="\r",
        )
        
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: #If user clicked close
                break
            elif event.type == pygame.JOYBUTTONDOWN:
                print("Joystick button pressed")
            elif event.type == pygame.JOYBUTTONUP:
                print("Joystick button released")

        if not paused:
            joystick_count = pygame.joystick.get_count()

            print("Number of joysticks: {}".format(joystick_count))

            
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        
            print("Joystick {}".format(0))

        
            name = joystick.get_name()
            print("Joystick name: {}".format(name))
        
            axes = joystick.get_numaxes()
            print("Number of axes: {}".format(axes))

        
            axis_0 = joystick.get_axis(0)#steering
            print("Axis {} value: {:>6.3f}".format(0, axis_0))

            #axis_1 = joystick.get_axis(1)
            #print("Axis {} value: {:>6.3f}".format(1, axis_1))
        
            #axis_2 = joystick.get_axis(2)#brake
            #print("Axis {} value: {:>6.3f}".format(2, axis_2))
    
            #axis_3 = joystick.get_axis(3)
            #print("Axis {} value: {:>6.3f}".format(3, axis_3))
    
            #axis_4 = joystick.get_axis(4)
            #print("Axis {} value: {:>6.3f}".format(4, axis_4))
    
            axis_5 = joystick.get_axis(5)#throttle
            print("Axis {} value: {:>6.3f}".format(5, axis_5))
        #
            output = [axis_0, axis_5] #[steering, throttle]
            print(output)

        output = counter_coordinates(output)
            
        save_data(dir_path=output_dir, images=img_seq, y=output, number=number_of_files)
        number_of_files += 1



        keys = key_check()
        if "Q" in keys and "E" in keys:
            paused = True
            print("\nStopping...")
            stop_recording.set()
            th_seq.join()
            th_img.join()
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.getcwd(),
        help="Directory where the training data will be saved",
    )

    parser.add_argument(
        "--save_everything",
        action="store_true",
        help="If this flag is added we will save every recorded sequence,"
        " it will result in a very unbalanced dataset. If this flag "
        "is not added we will use probability to try to generate a balanced "
        "dataset",
    )

    args = parser.parse_args()

    screen_recorder.initialize_global_variables()

    generate_dataset(
        output_dir=cartella, use_probability=not args.save_everything,
    )
