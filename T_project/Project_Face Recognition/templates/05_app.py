
from flask_wtf import Form
from wtforms import *
from wtforms.validators import *
from flask import Flask, render_template
from flask import url_for,render_template,request

app = Flask(__name__)

app.config['SECRET_KEY'] = ''

class ProgramForm(FlaskForm):
    name = StringField('What is Programming Language?', validatiors=[Required()])
    submit = SubmitField('Submit')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = ProgramForm()

    if form.validate_on_submit() == True:
        name = form.name.data
        form.name.data =''

    return render_template('index.html', form=form, name=name)


