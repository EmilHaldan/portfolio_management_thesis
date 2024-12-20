@echo off
cd "C:\Users\emilh\OneDrive\Dokumenter\Skole\Universitet (ITU)\11th Semester\Thesis\CodeBase"

echo     [%date% %time%]     Running delete_entire_db.py
python delete_entire_db.py

echo     [%date% %time%]     Running create_financial_database.py
python create_financial_database.py

echo     [%date% %time%]     Running create_financial_features.py
python create_financial_features.py

echo     [%date% %time%]     Running create_feature_data.py
python create_feature_data.py

echo     [%date% %time%]     Running create_ML_feature_dataset.py
python create_ML_feature_dataset.py

echo     [%date% %time%]     Running create_target_data.py
python create_target_data.py

echo     [%date% %time%]     Running create_edge_list.py
python create_edge_list.py

echo     [%date% %time%]     Running create_graph_data.py
python create_graph_data.py

echo     [%date% %time%]     Running create_graph_data.py
python create_LSTM_data.py



echo Data processing pipeline completed.
pause

