#     # Create a virtual environment
# python -m venv tts_project

# # Activate the virtual environment
# # On Windows:
# tts_project/Scripts/activate
# # On Linux/macOS:
# source tts_project/bin/activate
print("hello")

#https://huggingface.co/coqui/XTTS-v2/tree/main
# python version 3.9.6
# python --version
# python -m venv venv
# venv/Scripts/activate
# python.exe -m pip install --upgrade pip
# pip install TTS --cache-dir "C:/Users/DELL/Desktop/NEXTGENAI_Projects/Text-to-Speech Project with XTTS Model/text-to-speech-project-with-xtts-model/.cache"
# build tools Visual C++
# pip uninstall torch torchvision torchaudio
# pip install transformers datasets torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --cache-dir "C:/Users/DELL/Desktop/NEXTGENAI_Projects/Text-to-Speech Project with XTTS Model/text-to-speech-project-with-xtts-model/.cache"
# ##########################################################
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import soundfile as sf  # To save the output as a wav file
# import torch

# # Step 1: Load the model configuration
# config = XttsConfig()
# config.load_json("C:/Users/DELL/Desktop/NEXTGENAI_Projects/Text-to-Speech Project with XTTS Model/text-to-speech-project-with-xtts-model/assets/tts_config/config.json")

# # Step 2: Initialize the model
# model = Xtts.init_from_config(config)

# # Step 3: Load the pre-trained weights
# # model.load_checkpoint(config, checkpoint_dir="C:/Users/DELL/Desktop/NEXTGENAI_Projects/Text-to-Speech Project with XTTS Model/text-to-speech-project-with-xtts-model/assets/tts_config", eval=True)
# checkpoint_path = "C:/Users/DELL/Desktop/NEXTGENAI_Projects/Text-to-Speech Project with XTTS Model/text-to-speech-project-with-xtts-model/assets/tts_config/model.pth"
# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
# model.load_state_dict(checkpoint)

# # Optional: If you have CUDA installed and want to use GPU, uncomment the line below
# # model.cuda()

# def convertTTS(text, output_file):
#     # Step 4: Synthesize the output
#     outputs = model.synthesize(
#         "My name is chetana.",
#         config,
#         speaker_wav=r"C:\Users\DELL\Desktop\NEXTGENAI_Projects\Text-to-Speech Project with XTTS Model\text-to-speech-project-with-xtts-model\girl.wav",  # Replace with the correct path
#         gpt_cond_len=3,
#         language="en",
#     )

#     # Step 5: Save the synthesized speech to a wav file
#     output_wav = outputs['wav']
#     sf.write(str(output_file) + '.wav', output_wav, config.audio.sample_rate)

#     print("Speech synthesis complete and saved to output.wav")
###############################################################
# import soundfile as sf  # To save the output as a wav file
# from gtts import gTTS
# import os
# from pydub import AudioSegment



# def convertTTS(text, output_file):
#     # Step 1: Create a gTTS object
#     tts = gTTS(text=text, lang='en', slow=False)  # Change 'en' to the desired language code if needed
    
#     # Step 2: Save the synthesized speech to an MP3 file
#     output_mp3 = output_file + '.mp3'  # Saving as MP3 first
#     tts.save(output_mp3)

#     # Step 3: Convert MP3 to WAV using pydub
#     try:
#         # Load the MP3 file and convert to WAV
#         sound = AudioSegment.from_mp3(output_mp3)
#         sound.export(output_file + '.wav', format='wav')

#         # Optionally delete the MP3 file after conversion
#         os.remove(output_mp3)

#         print(f"Speech synthesis complete and saved to {output_file}.wav")
#     except Exception as e:
#         print(f"An error occurred during the conversion: {e}")

# # Example usage
# text = "नमस्ते! यह gTTS का उपयोग करके एक नमूना पाठ से भाषण रूपांतरण है। मेरा नाम चेतना है। मैं जनरेटिव एआई सीख रही हूँ।"
# convertTTS(text, "output")

#############################################################################################################

#converting and translating
import soundfile as sf  # To save the output as a wav file
from gtts import gTTS
import os
from pydub import AudioSegment
from googletrans import Translator  # Import the translator

def convertTTS(text, output_file):
    # Step 1: Initialize the translator
    translator = Translator()
    
    # Step 2: Translate the Hindi text to English
    translated_text = translator.translate(text, src='hi', dest='en').text

    # Step 3: Create a gTTS object with the translated text
    tts = gTTS(text=translated_text, lang='en', slow=False)  # Using English language code
    
    # Step 4: Save the synthesized speech to an MP3 file
    output_mp3 = output_file + '.mp3'  # Saving as MP3 first
    tts.save(output_mp3)

    # Step 5: Convert MP3 to WAV using pydub
    try:
        # Load the MP3 file and convert to WAV
        sound = AudioSegment.from_mp3(output_mp3)
        sound.export(output_file + '.wav', format='wav')

        # Optionally delete the MP3 file after conversion
        os.remove(output_mp3)

        print(f"Speech synthesis complete and saved to {output_file}.wav")
    except Exception as e:
        print(f"An error occurred during the conversion: {e}")

# Example usage
text_hindi = "नमस्ते! यह gTTS का उपयोग करके एक नमूना पाठ से भाषण रूपांतरण है। मेरा नाम चेतना है। मैं जनरेटिव एआई सीख रही हूँ।"
convertTTS(text_hindi, "output")
