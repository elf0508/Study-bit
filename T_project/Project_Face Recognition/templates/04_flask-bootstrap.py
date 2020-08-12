from flask import Flask, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

@app.route('/program/<name>')
def program(name):
    return render_template('program.html', name=name)

if __name__=='__main__':
    app.run()
    