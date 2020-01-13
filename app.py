from flask import Flask, render_template
import datetime
from cam import Cam

app = Flask(__name__)
cam = Cam()

@app.route("/")
def index():
   return render_template('index.html')

@app.route("/snap")
def snap():
   cam.snap()
   return 'ok';

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=80, debug=True)