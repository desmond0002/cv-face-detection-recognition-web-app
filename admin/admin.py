from flask import Blueprint, render_template, session, redirect, request, url_for, flash
from face_extractor import  extract_faces
from source.model_training import train_svm_model, train_nb_model, train_xgboost_model
from test import make_list
from source.embedding_extraction import extract_embeddings
#from app import isLogged

admin = Blueprint('admin', __name__, template_folder='templates', static_folder='static')

# if current_user.get_id() == user.id:
#     login_admin()

def login_admin():
    session['admin_logged'] = 1

def logout_admin():
    session.pop('admin_logged', None)

def isLogged():
    return True if session.get('admin_logged') else False

@admin.route('/', methods=['GET'])
def admin_page():
    if isLogged():
        return render_template('admin/admin.html')
    else:
        return render_template('main.html')

@admin.route('/login', methods=['POST', 'GET'])
def admin_login():
    if isLogged():
        return render_template('admin/admin.html')
    if request.method == 'POST':
        if request.form['login'] == 'admin' and request.form['password'] == 'admin':
            login_admin()
            return redirect(url_for('.admin_page'))
        else:
            flash('Неверная пара логин/пароль', "error")
    return  render_template('admin/login.html')

@admin.route('/logout', methods=['POST', 'GET'])
def admin_logout():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    logout_admin()
    return render_template('main.html')

@admin.route('/prep', methods=['POST', 'GET'])
def prep():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/prep.html')

@admin.route('/prep_embeddings', methods=['POST', 'GET'])
def prep_embeddings():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    classes_dir = request.form.get('classes_dir')
    embeddings_path = request.form.get('embeddings_path')
    extract_embeddings(classes_dir, embeddings_path)
    return render_template('admin/prep.html')

@admin.route('/train', methods=['POST', 'GET'])
def train():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/train.html')

@admin.route('/test', methods=['POST', 'GET'])
def test():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/test.html')

@admin.route('/extract', methods=['GET', 'POST'])
def extract_faces_btn():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    path = request.form.get('extract_path')
    extract_faces(path)
    return render_template('admin/prep.html')

@admin.route('/train_svm', methods=['GET', 'POST'])
def train_svm():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/train_svm.html')

@admin.route('/train_nb', methods=['GET', 'POST'])
def train_nb():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/train_nb.html')

@admin.route('/train_xgb', methods=['GET', 'POST'])
def train_xgb():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    return render_template('admin/train_xgb.html')

@admin.route('/train_svm_classifier', methods=['GET', 'POST'])
def train_svm_classifier():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    embeddings_path = request.form.get('embeddings_path')
    classifier_model_path = request.form.get('classifier_model_path')
    label_encoder_path = request.form.get('label_encoder_path')
    # print(embeddings_path)
    # print(classifier_model_path)
    # print(label_encoder_path)
    train_svm_model(embeddings_path, classifier_model_path, label_encoder_path)
    return render_template('admin/train_svm.html')

@admin.route('/train_nb_classifier', methods=['GET', 'POST'])
def train_nb_classifier():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    embeddings_path = request.form.get('embeddings_path')
    classifier_model_path = request.form.get('classifier_model_path')
    label_encoder_path = request.form.get('label_encoder_path')
    train_nb_model(embeddings_path, classifier_model_path, label_encoder_path)
    return render_template('admin/train_nb.html')

@admin.route('/train_xgboost_classifier', methods=['GET', 'POST'])
def train_xgboost_classifier():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    embeddings_path = request.form.get('embeddings_path')
    classifier_model_path = request.form.get('classifier_model_path')
    label_encoder_path = request.form.get('label_encoder_path')
    train_xgboost_model(embeddings_path, classifier_model_path, label_encoder_path)
    return render_template('admin/train_xgb.html')

@admin.route('/test_images', methods=['GET', 'POST'])
def test_images():
    if not isLogged():
        return redirect(url_for('.admin_login'))
    classes_dir = request.form.get('classes_dir')
    classifier_model_path = request.form.get('classifier_model_path')
    label_encoder_path = request.form.get('label_encoder_path')
    make_list(classes_dir, classifier_model_path, label_encoder_path)
    return render_template('admin/test.html')