from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX = int(request.form.get('SEX')),
            EDUCATION = int(request.form.get('EDUCATION')),
            MARRIAGE = int(request.form.get('MARRIAGE')),
            AGE = float(request.form.get('AGE')),
            PAY_1 = int(request.form.get('PAY_1')),
            BILL_AMT1 = float(request.form.get('BILL_AMT1')),
            PAY_AMT1= float(request.form.get('PAY_AMT1')),
            PAY_AMT2 = float(request.form.get('PAY_AMT2')),
            PAY_AMT3 = float(request.form.get('PAY_AMT3')),
            PAY_AMT4 = float(request.form.get('PAY_AMT4')),
            PAY_AMT5 = float(request.form.get('PAY_AMT5')),
            PAY_AMT6 = float(request.form.get('PAY_AMT6'))  
        )
        final_new_data=data.get_data_as_dataframe()
        #print(f"{final_new_data}")
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        print(pred)
        results = pred[0]
        if (results==1):
            results = "Defaulter"
        else:
            results = "Not Defaulter"
            
        return render_template('index.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

