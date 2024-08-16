# LLM Watermarking Framework and Toolkit

## Overview
This repository contains the open-source toolkit and research materials for the "Credibility-Driven Multi-bit Watermark for Large Language Models Identification" project. The project introduces a novel multi-party credible watermarking framework that enhances the identification, privacy, and security of Large Language Models (LLMs). The watermarking scheme is designed to embed and extract multi-bit information without compromising the quality of the generated text.

## Key Features
- **Multi-party Participation**: A collaborative watermarking framework involving a Trusted Third Party (TTP) and multiple LLM vendors.
- **Privacy Protection**: Ensures that user prompts and model outputs remain confidential during the watermarking process.
- **Flexibility and Scalability**: Easily accommodates different vendors' private watermarking technologies and allows for the upgrading of authentication and encryption algorithms.
- **Robust Watermarking Algorithm**: A multi-bit watermarking method that improves success rates and information capacity, with a focus on maintaining text quality.
- **Open-source Toolkit**: A user-friendly toolkit that includes watermarking frameworks, experiment configurations, datasets, and baseline algorithms.

## Repository Structure
```
llm-watermarking/
│
├── paper/                  # Research paper and related documents
│   ├── Credibility-Driven_Multi-bit_Watermark.pdf
│   └── Reproducibility_Checklist.md
│
├── config/                 # Configuration files for various watermark algorithms          
│   ├── CredID.json       
│   ├── EXP.json           
│   ├── KGW.json
│   ├── CTWL.json            
│   ├── SIR.json            
│   ├── SWEET.json         
│   ├── Unigram.json        
│   ├── MPAC.json           
│   └── Cyclic-shift.json    
│       
├── src/                    # Source code for the watermarking framework
│   ├── embedding/          # Embedding watermark into LLM responses
│   ├── extraction/         # Extracting watermark from text
│   └── utils/              # Utility functions and modules
│
├── datasets/               # Sample datasets used in the experiments
│   ├── OpenGen/
│   ├── C4_News/
│   └── Essays/
│
├── watermarkings/          #Implementation framework for watermark algorithms
│   ├── watermark_processor.py
│   ├── base.py
│   ├── KGW/
│   ├── EXP/
│   ├── SIR/
│   ├── SWEET/
│   ├── Unigram/
│   ├── CredID/
│   ├── CTWL/
│   ├── MPAC/
│   └── Cyclic-shift/
│
├── attacks/                #Implementation framework for watermark algorithms
│   ├── get_datasets.py
│   ├── cp_attack.py
│   ├── Deletion_attack.py/
│   ├── Substitution_attack.py
│   └── Homoglyph_attack.py
│
├── evaluation/                 # Evaluation module, including tools and pipelines
│   ├── dataset.py              # Script for handling dataset operations within evaluations
│   ├── examples/               # Scripts for automated evaluations using pipelines
│   │   ├── assess_success_rate.py  
│   │   ├── assess_quality.py
│   │   ├── assess_speed.py    
│   │   └── assess_robustness.py   
│   ├── pipelines/              # Pipelines for structured evaluation processes
│   │   ├── speed_analysis.py    
│   │   ├── success_rate_analysis.py    
│   │   └── quality_analysis.py 
│   └── tools/                  # Evaluation tools
│       ├── ppl.py
│       ├── success_rate_calculator.py  
│   │   └── BLEU.py   
│
├── demo/                   # Credible watermark framework cases and examples for user testing
│   ├── test_method_single_party.py      
│   ├── test_pipeline_multi_party.py    
│   └── test_real_word.py
│
├── experiments/            # Scripts and configurations for running experiments
│
├── README.md               # Main project documentation
└── requirements.txt        # Dependencies required for the project
```

## Installation
To install the required dependencies and set up the environment, follow these steps:
1. Clone the repository:
   ```
   git clone https://anonymous.4open.science/r/credible-LLM-watermarking-7D62.git
   ```
2. Navigate to the repository:
   ```
   cd credible-LLM-watermarking
   ```
3. Create a virtual environment and activate it (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
The toolkit provides functionalities for both watermark embedding and extraction. Here are the basic steps to use the toolkit:

### Embedding Watermarks
1. Prepare your LLM and the text prompt.
2. Use the embedding module to integrate the watermark into the LLM's response.

### Extracting Watermarks
1. Obtain the watermarked text.
2. Utilize the extraction module to decode and verify the watermark.

### Running Experiments
1. Configure the experiment settings in the `experiments/` directory.
2. Run the experiment scripts to evaluate the watermarking performance.

## Contributing
We welcome contributions to the project. Please follow the guidelines in `CONTRIBUTING.md` for submitting issues and pull requests.

## License
This project is licensed under the [LICENSE_NAME](LICENSE). Please refer to the license file for more information.

## Citation
If you use this toolkit in your research, please cite our paper:
```
@misc{anonymous2024watermarking,
  author = {Anonymous},
  title = {Credibility-Driven Multi-bit Watermark for Large Language Models Identification},
  year = {2024},
  howpublished = {https://anonymous.4open.science/r/credible-LLM-watermarking-7D62},
}
```

## Acknowledgements
We would like to acknowledge the contributions of the LLM research community and the developers of the open-source models used in our experiments.

---

Please replace `LICENSE_NAME` with the actual name of the license you are using for this project. If you need any further customization or have specific requirements, feel free to ask.
