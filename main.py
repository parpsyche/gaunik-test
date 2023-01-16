from flask import Flask, request,jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
@app.route('/',methods =['GET'])
async def hello():
    return 'hello'

@app.route('/predict', methods=['POST'])
async def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    '''
    fruits_veggies = int(request.form.get('FRUITS_VEGGIES'))
    places_visited = int(request.form.get('PLACES_VISITED'))
    core_circle = int(request.form.get('CORE_CIRCLE'))
    supporting_others = int(request.form.get('SUPPORTING_OTHERS'))
    achievement = int(request.form.get('ACHIEVEMENT'))
    donation = int(request.form.get('DONATION'))
    bmi_range = int(request.form.get('BMI_RANGE'))
    todo_completed = int(request.form.get('TODO_COMPLETED'))
    flow = int(request.form.get('FLOW'))
    daily_steps = int(request.form.get('DAILY_STEPS'))
    live_vision = int(request.form.get('LIVE_VISION'))
    sleep_hours = int(request.form.get('SLEEP_HOURS'))
    lost_vacation = int(request.form.get('LOST_VACATION'))
    daily_shouting = int(request.form.get('DAILY_SHOUTING'))
    sufficient_income = int(request.form.get('SUFFICIENT_INCOME'))
    personal_awards = int(request.form.get('PERSONAL_AWARDS'))
    time_for_passion = int(request.form.get('TIME_FOR_PASSION'))
    weekly_meditation = int(request.form.get('WEEKLY_MEDITATION'))
    age = int(request.form.get('AGE'))
    gender = int(request.form.get('GENDER'))
    wlbs = int(request.form.get('WORK_LIFE_BALANCE_SCORE'))

    

    pred_name = np.array([[fruits_veggies,places_visited,core_circle,supporting_others,achievement,
    donation,bmi_range,todo_completed,flow,daily_steps,live_vision,sleep_hours,lost_vacation,
    daily_shouting,sufficient_income,personal_awards,time_for_passion,weekly_meditation,
    age, gender,wlbs]])
    '''
    score = model.predict(query_df)

    return jsonify({'Prediction':int(score)})


if __name__ == '__main__':
    app.run(debug=True)
