import matplotlib.pyplot as plt
def plot_target(dataset):
    
    #Fraud and non fraud cases count in pie chart for the dataset
    round(100*dataset['FLAG'].value_counts(normalize=True),2).plot(kind='pie',explode=[0.02]*2, figsize=(6, 6), autopct='%1.2f%%')
    plt.title("Fraudulent and Non-Fraudulent Distribution")
    plt.legend(["Non-Fraud", "Fraud"])
    plt.show()
    

