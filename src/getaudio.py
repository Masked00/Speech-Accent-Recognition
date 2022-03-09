import pandas as pd
import urllib.request
import os
import sys
from pydub import AudioSegment
from colorama import init, Fore, Style

init(autoreset=True)
red = Fore.RED + Style.BRIGHT
green = Fore.GREEN + Style.BRIGHT
yellow = Fore.YELLOW + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT
magenta = Fore.MAGENTA + Style.BRIGHT
cyan = Fore.CYAN + Style.BRIGHT
white = Fore.WHITE + Style.BRIGHT


class GetAudio:

    def __init__(self, csv_filepath, destination_folder='audio/', wait=0, debug=False):
       
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.url = 'http://chnm.gmu.edu/accent/soundtracks/{}.mp3'
        self.destination_folder = destination_folder
        self.wait = wait
        self.debug = True

    def check_path(self):
       
        if not os.path.exists(self.destination_folder):
            if self.debug:
                print('{} does not exist, creating'.format(
                    self.destination_folder))
            os.makedirs('../' + self.destination_folder)

    def get_audio(self):
        

        self.check_path()

        counter = 0

        for lang_num in self.audio_df['language_num']:
            if not os.path.exists(self.destination_folder + '{}.wav'.format(lang_num)):
                if self.debug:
                    print(f'  {white}> {magenta}downloading {white}{lang_num}')
                (filename, headers) = urllib.request.urlretrieve(
                    self.url.format(lang_num))
                sound = AudioSegment.from_mp3(filename)
                sound.export('../' + self.destination_folder +
                             "{}.wav".format(lang_num), format="wav")
                counter += 1

        return counter


if __name__ == '__main__':
    
    csv_file = sys.argv[1]
    ga = GetAudio(csv_filepath=csv_file)
    ga.get_audio()
