import collections

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False

class HeightObject:
	def __init__(self, objectID, height):
		# store the object ID, then initialize a list of heights
		self.objectID = objectID
		self.heights = [height]

		self.height = None

	def determine_height(self):
		#update the height with mean value
		self.height = sum(self.heights)/len(self.heights)

class GenderObject:
	def __init__(self, objectID, gender):
		# store object ID, then initialize a list of gender
		self.objectID = objectID
		self.genders = [gender]

		self.gender = None
	
	def determine_gender(self):
		data = collections.Counter(self.genders)
		data_list = dict(data)
		max_val = max(list(data.values()))
		mode_val = [num for num, freq in data_list.items() if freq == max_val]
		self.gender = mode_val[0]