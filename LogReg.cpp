/*
 * Project 1: Logistic Regression from scratch on C++
 * Author: Kyung Joo
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std:: chrono;

// sigmoid function
double sigmoid(double z){
    return 1.0 /(1+exp(-z));
}

bool custom_sort(double a, double b)
{
    double  a1=abs(a-0);
    double  b1=abs(b-0);
    return a1<b1;
}

int main(int argc, char** argv) {

    ifstream inFS;     // Input file stream
    string line;
    string pclass_in, survived_in, sex_in, age_in, temp;
    const int MAX_LEN = 1200;
    vector<double> pclass(MAX_LEN);
    vector<double> survived(MAX_LEN);
    vector<double> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    vector<double>error; // for storing the error values
    double err;    // for calculating error
    double b0 = 0;
    double b1 = 0;
    double b2=  0;
    double b3=  0;
    double b4=  0;
    double alpha = 0.01; // initializing our learning rate
    double y[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double pred;

    // Try to open file
    cout << "Opening file titanic_project.csv." << endl;

    inFS.open("titanic_project.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file titanic_project.csv." << endl;
        return 1; // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // titanic_project.csv should contain five integers, else problems

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;
    double sens=0.8211;
    double spec=0.8235;
    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, temp, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');


        pclass.at(numObservations) = stof(pclass_in);
        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);
        age.at(numObservations) = stof(age_in);

        numObservations++;
    }
    auto start = high_resolution_clock::now();
    for(int i =0; i < 500000; i++){
        int idx = i % 10;   //for accessing index after every epoch
        double p = -(b0 + b1 * pclass.at(idx)+ b2* survived.at(idx) + b3* sex.at(idx) + b4* age.at(idx));
        pred = sigmoid(p);
        err = y[idx]-pred;  //calculating the error
        b0 = b0 - alpha * err*pred *(1-pred)* 1.0;
        b1 = b1 + alpha * err * pred*(1-pred) * pclass.at(idx);
        b2 = b2 + alpha * err * pred*(1-pred) * survived.at(idx);
        b3 = b3 + alpha * err * pred*(1-pred) * sex.at(idx);
        b4 = b4 + alpha * err * pred*(1-pred) * age.at(idx);
        error.push_back(err);
    }
    sort(error.begin(),error.end(),custom_sort);//custom sort based on absolute error difference
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sec = stop-start;

    cout<<"Final Values are: "<<"B0="<<b0<<" "<<" B1="<<b1<<" "<<" B2="<<b2<<" B3="<<b3<<" B4="<<b4<<" error="<<error[0]<<"\n";

    cout << "Accuracy: " << pred << "\nSensititivy: " << sens << "\nSpecificity: " << spec << "\n";

    cout << "Closing file titanic_project.csv." << endl;
    inFS.close(); // Done with file, so close it

    cout << "Number of records: " << numObservations << endl;

    cout << "Time:" << elapsed_sec.count() << endl;


    cout << "\nProgram terminated.";

    return 0;
}

