import os
import csv

dataset = 'm1'
source_dir = 'D:\COVID-Net' + '\\' + dataset + '\\'
bucket_url = 'gs://xrteam_c19'

# gsutil -m cp -r m1 gs://xrteam_c19

def generate_cvs_file(rootdir, output_csv_file_name):
    with open(output_csv_file_name, mode='w', newline='') as training_file:
        training_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                subdir_split = subdir.split("\\")
                instance_class = subdir_split[-1]
                file_full_path = bucket_url + '/' + dataset + '/' + instance_class + '/' + file
                training_writer.writerow([file_full_path,instance_class])

if __name__ == '__main__':
    csv_file_name = dataset + '_bucket.csv'
    csv_file_name_final = csv_file_name.replace(" ","")
    generate_cvs_file(source_dir, csv_file_name_final)
