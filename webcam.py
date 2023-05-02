import cv2
import os
from google.cloud import vision
import io
from google.oauth2.service_account import Credentials
import pafy ## also install pip install git+https://github.com/ytdl-org/youtube-dl.git@master#egg=youtube_dl


SCOPES = ['https://www.googleapis.com/auth/classroom.courses.readonly']
credentials = Credentials.from_service_account_file('facialdetection-384720-04866abf3cf6.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

## youtube url
# url="https://www.youtube.com/watch?v=BS9gX34VpkI"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
  
def func():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def detectFacialExpressionByURL(url):
    # Capture video from Youtube URL
    cap = cv2.VideoCapture(url)

    while True:
        # Read frame from video stream
        ret, frame = cap.read()
        if not ret:
        # Exit loop if there are no more frames
            break
        # Detect faces in the frame using GCP Vision API
        image = vision.Image(content=cv2.imencode('.jpg', frame)[1].tobytes())
        response = client.face_detection(image=image)
        faces = response.face_annotations
        
        # Analyze facial expressions for each detected face
        for face in faces:
            emotions = {}
            if face.joy_likelihood != vision.Likelihood.UNKNOWN:
                emotions['joy'] = face.joy_likelihood
            if face.sorrow_likelihood != vision.Likelihood.UNKNOWN:
                emotions['sorrow'] = face.sorrow_likelihood
            if face.anger_likelihood != vision.Likelihood.UNKNOWN:
                emotions['anger'] = face.anger_likelihood
            if face.surprise_likelihood != vision.Likelihood.UNKNOWN:
                emotions['surprise'] = face.surprise_likelihood
            if face.under_exposed_likelihood != vision.Likelihood.UNKNOWN:
                emotions['under_exposed'] = face.under_exposed_likelihood
            if emotions:
                # Determine dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                print(dominant_emotion)
            
        # Display video stream with face detection
        cv2.imshow('Real-time Face Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def detect_faces(path):
    """Detects faces in an image."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in face.bounding_poly.vertices])

        # print('face bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


if __name__=="__main__":
    print(" main --")
    ## youtube urlid 
    ## suprise JNQU-4YEnm4
    ## crying 30SRQ5RVQiM
    url="30SRQ5RVQiM"
    video = pafy.new(url,basic=False)
    best = video.getbestvideo(preftype="mp4")
    print(best.url)
    detectFacialExpressionByURL(best.url)

    