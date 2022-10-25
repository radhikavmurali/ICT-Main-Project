from flask import Flask,render_template,request
import numpy as np
import pickle
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
product_category_name=pickle.load(open('product_category_name.pkl','rb'))
seller_city=pickle.load(open('seller_city.pkl','rb'))
seller_state=pickle.load(open('seller_state.pkl','rb'))
order_status=pickle.load(open('order_status.pkl','rb'))
payment_type=pickle.load(open('payment_type.pkl','rb'))
customer_city=pickle.load(open('customer_city.pkl','rb'))
customer_state=pickle.load(open('customer_state.pkl','rb'))



@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        price	= float(request.form['price'])
        freight_value=float(request.form['freight_value'])
        product_category_name=request.form['product_category_name']
        product_name_lenght=float(request.form['product_name_lenght'])
        product_description_lenght= float(request.form['product_description_lenght'])
        product_photos_qty=float(request.form['product_photos_qty'])
        product_weight_g=float(request.form['product_weight_g'])                       
        seller_city =request.form['seller_city']
        seller_state=request.form['seller_state']
        order_status=request.form['order_status']
        payment_type=request.form['payment_type']
        payment_installments= float(request.form['payment_installments'])
        payment_value=float(request.form['payment_value'])
        customer_city=request.form['customer_city']
        customer_state=request.form['customer_state']
        estimated_time=float(request.form['estimated_time']) 
        actual_time	= float(request.form['actual_time'])
        diff_actual_estimated=float(request.form['diff_actual_estimated'])
        diff_approval_shipping=float(request.form['diff_approval_shipping'])
        shipping_time=float(request.form['shipping_time'])
        product_size= float(request.form['product_size'])
        
    product_category_name=label_encoder.fit_transform([product_category_name])
    seller_city=label_encoder.fit_transform([seller_city])
    seller_state=label_encoder.fit_transform([product_category_name])
    order_status=label_encoder.fit_transform([order_status])
    payment_type=label_encoder.fit_transform([payment_type])
    customer_city=label_encoder.fit_transform([customer_city])
    customer_state=label_encoder.fit_transform([customer_state])
    
    
    details=[price,freight_value,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,seller_city,seller_state,order_status,payment_type,payment_installments,payment_value,customer_city,customer_state,estimated_time,actual_time,diff_actual_estimated,diff_approval_shipping,shipping_time,product_size]
    data_out=np.array(details).reshape(1,-1)
    output=model.predict(data_out)
    output=output.item()
    return render_template('result.html',data=output)
   # return render_template('result.html',prediction_text='Customer is  {}'.format(output)) 
       


if __name__ == '__main__':
    app.run(port=5000)