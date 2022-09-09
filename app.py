import os
from flask import Flask, jsonify, request, render_template, url_for, flash, redirect, session
from source.face_recognition import recognize_faces
from source.utils import draw_rectangles, read_image, prepare_image
from source.face_detection import detect_faces_with_ssd
from flask_login import LoginManager, login_manager, login_user, logout_user, UserMixin, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from flask_sqlalchemy import SQLAlchemy
from admin.admin import admin


# @login_required
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/andrew/flask_db.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.register_blueprint(admin, url_prefix='/admin')
db = SQLAlchemy(app)
manager = LoginManager(app)
app.secret_key = 'qwerty321'
db.create_all()
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER





@manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    login = db.Column(db.String(128), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)


@app.route('/', methods=['GET'])
def begin():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    login = request.form.get('login')
    password = request.form.get('password')
    password2 = request.form.get('password2')

    if request.method == 'POST':
        if not (login or password or password2):
            flash('Пожалуйста, заполните все поля')
        elif password != password2:
            flash('Пароли не совпадают')
        else:
            hash_pwd = generate_password_hash(password)
            new_user = User(login=login, password=hash_pwd)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))

    return render_template('Register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    login = request.form.get('login')
    password = request.form.get('password')

    if login and password:
        user = User.query.filter_by(login=login).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            #next_page = request.args.get('next')
            return redirect('home')
        else:
            flash('Неправильный логин или пароль!')
    else:
        flash('Заполните поля логина и пароля')
    return render_template('login.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('begin'))


@app.after_request
def redirect_to_signin(response):
    if response.status_code == 401:
        return redirect(url_for('login')  + '?next=' + request.url)
    return response


@app.route('/home')
@login_required
def home():
    return render_template('main.html')


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)
    # Recognize faces
    faces = detect_faces_with_ssd(image)
    # return requests.get(url).json()
    return jsonify(detections=faces)


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)

    # Recognize faces
    classifier_model_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "newrec20.pickle"
    label_encoder_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "newlab20.pickle"

    faces = recognize_faces(image, classifier_model_path, label_encoder_path)

    # Draw detection rects
    draw_rectangles(image, faces)

    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('main.html', face_recognized=len(faces) > 0, num_faces=len(faces), image_to_show=to_send,
                           init=True)


if __name__ == '__main__':
    app.run(debug=True,
            use_reloader=True,
            port=5000)
