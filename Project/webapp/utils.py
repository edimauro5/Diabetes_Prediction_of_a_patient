from matplotlib import pyplot as plt
from numpy import concatenate, save
from os import environ, makedirs, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

'''
This function creates the path in which to save a plot (plots are shown), a numpy array or a model and saves that file
'''
def save_file(dir_path, file_name, file = None):
    if not path.exists(dir_path):
        makedirs(dir_path)
    if file is None:
        plt.savefig(path.join(dir_path, file_name), dpi = 300, bbox_inches = 'tight')
        plt.show()
    elif file_name.split('.')[-1] == 'h5':
        keras.models.save_model(file, path.join(dir_path, file_name))
    elif file_name.split('.')[-1] == 'npy':
        save(path.join(dir_path, file_name), file)

'''
This function takes in input the reference values and the prediction values as lists, returns a list with each index corresponding to the total number
of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and creates the plot of clarke_error_grid
'''
def clarke_error_grid(ref_values, pred_values, title_string):
    #Check to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), 'Unequal number of values (reference : {}) (prediction : {}).'.format(len(ref_values), len(pred_values))

    #Check to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print('Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).'.format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print('Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.'.format(min(ref_values),  min(pred_values)))

    #Clear plot
    plt.clf()
    plt.figure(figsize=(6, 6))
    
    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
    plt.title(title_string)
    plt.xlabel('Reference Concentration (mg/dl)')
    plt.ylabel('Prediction Concentration (mg/dl)')
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    #Add zones titles
    plt.text(30, 15, 'A', fontsize=15)
    plt.text(370, 260, 'B', fontsize=15)
    plt.text(280, 370, 'B', fontsize=15)
    plt.text(160, 370, 'C', fontsize=15)
    plt.text(160, 15, 'C', fontsize=15)
    plt.text(30, 140, 'D', fontsize=15)
    plt.text(370, 120, 'D', fontsize=15)
    plt.text(30, 370, 'E', fontsize=15)
    plt.text(370, 15, 'E', fontsize=15)

    #Statistics from the data
    zones = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zones[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zones[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zones[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zones[3] += 1    #Zone D
        else:
            zones[1] += 1    #Zone B
    zones = [zone/len(ref_values) for zone in zones]
    
    return zones

'''
This function creates the plot of real values vs predicted values
'''
def real_pred(real, predicted, title_string):
    plt.clf()
    plt.figure(figsize=(6, 6))
    
    plt.title(title_string)
    plt.xlabel('Time')
    plt.ylabel('Glucose')
    plt.plot(real, label = 'Real')
    plt.plot(predicted, label = 'Predicted')
    plt.legend()

'''
This function tests the LSTM model on the test set, plotting the prediction and the real values, calculating the evaluation
metrics (MAE, MSE, RMSE) and plotting the clarke error grid
'''
def test_model(test_X, test_y, model, patient_id):    
    # Make a Prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # Invert scaling for predicted
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis = 1)
    inv_yhat = inv_yhat[:, 0]

    # Invert scaling for real
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis = 1)
    inv_y = inv_y[:, 0]
    
    # Save and plot Predicted values vs Real values
    real_pred(inv_y, inv_yhat, f'Patient {patient_id} - Real vs Predicted')
    save_file(f'../pipeline/Patients_info/{patient_id}', f'{patient_id}_real_pred.png')
    
    # Save and plot Clarke error grid
    zones = clarke_error_grid(inv_y*180, inv_yhat*180, f'Patient {patient_id} - Clarke error grid')
    print('Clarke error grid zones')
    print(zones)
    save_file(f'../pipeline/Patients_info/{patient_id}', f'{patient_id}_clarke.png')
    
    return inv_y, inv_yhat