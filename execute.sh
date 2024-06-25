#!/bin/bash


##################################################
# Create table.
##################################################

export PGUSER=dwhuser
export PGPASSWORD=dwhuser

investment_objective_raw='sql/ddl/investment_objective_raw.sql'
psql -h localhost -d dwh -f "${investment_objective_raw}" >/dev/null


##################################################
# Document type: Prospectus.
##################################################

concatenated_output_file=data/prospectus_investment_objective.txt


while IFS= read -r -d '' target
do

  echo -n "Cleansing, splitting, and adding metadata to ${target} ... "

  # Extract metadata.
  fund_id=$(echo "${target}" | sed -r 's/^.*\/([A-Z0-9]{12})\/.*$/\1/')

  output_file=$(python bin/python/cleanse_text.py "${target}" -c '\t' -c '\n')
  output_file=$(python bin/python/split_text.py "${output_file}" -r)
  output_file=$(python bin/python/add_metadata.py "${output_file}" -m "${fund_id}" -m prospectus -r)

  echo "Success!"

  output_file=$(python bin/python/extract_vector.py "${output_file}" -s 3 -r)

done <   <(find data -name 'prospectus_investment_objective.txt' -print0)


while IFS= read -r -d '' target
do
  cat "${target}" >> "${concatenated_output_file}"
done <   <(find data -name 'vectorised_*_prospectus_investment_objective.txt' -print0)

# concatenated_output_file=$(python bin/python/compress_vector_tsne.py "${concatenated_output_file}" -v 4 -r)
# concatenated_output_file=$(python bin/python/compress_vector_clf.py "${concatenated_output_file}" -v 4 -r)
# concatenated_output_file=$(python bin/python/load_vector.py "${concatenated_output_file}" -t 'investment_objective_raw' -c 'fund_id,document_type,sentence_no,sentence,embedding,z0,z1' -o -r)



##################################################
# Run Jupyter Notebooks using configured conda environment
##################################################

echo "Running Jupyter Notebook: VAE_inference.ipynb ..."
conda_env_name='hand_obj_processing'
# Activate the Conda environment
source activate "${conda_env_name}"


##################################################
# Run Jupyter Notebook: VAE_train.ipynb
##################################################

# Path to the notebook
train_notebook_path="notebooks/VAE_train.ipynb"

# Output path for the executed notebook
executed_train_notebook_path="VAE_train_executed.ipynb"

# Execute the notebook and save the output
jupyter nbconvert --to notebook --execute "${train_notebook_path}" --output "${executed_train_notebook_path}"

if [ $? -eq 0 ]; then
  echo "Training Notebook executed successfully and saved as ${executed_train_notebook_path}."
else
  echo "Failed to execute the  training notebook."
fi


##################################################
# Run Jupyter Notebook: VAE_inference.ipynb
##################################################

# Path to the notebook
inference_notebook_path="notebooks/VAE_inference.ipynb"

# Output path for the executed notebook
executed_inference_notebook_path="VAE_inference_executed.ipynb"

# Execute the notebook and save the output
jupyter nbconvert --to notebook --execute "${inference_notebook_path}" --output "${executed_inference_notebook_path}"

if [ $? -eq 0 ]; then
  echo "Inference Notebook executed successfully and saved as ${executed_inference_notebook_path}."
else
  echo "Failed to execute the  inference notebook."
fi


conda deactivate