class Parameter(object):
	"""docstring for Parameter"""
	def __init__(self):
		super(Parameter, self).__init__()
		self.value = None
		self.stop = None

	def get_value(self):
		if self.value is None:
			self.value = 0
		return self.value

	def requires_stop(self):
		return self.stop
		