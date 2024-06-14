import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Ensure the necessary directories exist
os.makedirs('user_data', exist_ok=True)
os.makedirs('login_data', exist_ok=True)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    auth = data.get('auth')
    name = data.get('name')
    ID = data.get('ID')
    password = data.get('password')
    patient_id = data.get('patient_id')

    # Check for ID duplication
    if os.path.exists(f'login_data/{ID}.txt'):
        return '', 400

    if auth == 1:
        # Check if patient file exists
        if os.path.exists(f'user_data/{patient_id}.txt'):
            # Create login data file
            with open(f'login_data/{ID}.txt', 'w') as file:
                file.write(f'{password}\n{auth}\n{name}')
            return '', 200
        else:
            return '', 404
    elif auth in [2, 4]:
        # Create login data file without checking patient ID
        with open(f'login_data/{ID}.txt', 'w') as file:
            file.write(f'{password}\n{auth}\n{name}')
        return '', 200
    else:
        return '', 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    ID = data.get('ID')
    password = data.get('password')

    # Check if login data file exists
    if not os.path.exists(f'login_data/{ID}.txt'):
        return '', 400

    # Read the login data file
    with open(f'login_data/{ID}.txt', 'r') as file:
        stored_password = file.readline().strip()
        auth = file.readline().strip()
        name = file.readline().strip()

    if password == stored_password:
        return jsonify({'status_code': 200, 'auth': auth, 'name': name}), 200
    else:
        return '', 400

if __name__ == '__main__':
    app.run(debug=True)
