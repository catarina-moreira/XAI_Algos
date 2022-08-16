#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

class Constants(object):
	def __init__(self, dataset_name):
		self.MAIN_PATH : str = ""
		self.DATASET_PATH : str = os.path.join("", "data", "generated_datasets", dataset_name + ".csv")

