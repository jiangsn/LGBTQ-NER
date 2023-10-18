#!/bin/bash
source ~/.bashrc
cd /home/jian/mengling/TobaccoResearch/SpacyProdigy
prodigy ner.manual tobacco_ner_test1 blank:en tobacco_ner_example.jsonl --label D-rac,D-gen,B-ces,B-use,M-sts,P-reg,LOCATION > tobacco.log
