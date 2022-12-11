"""
Utilities, borrowed from Yolov7
"""

# Adapted version of yolov7's attempt_download
def attempt_download(file, repo='chrisgilldc/delv'):
	# Attempt file download if does not exist
	file = Path(str(file).strip().replace("'", '').lower())

	if not file.exists():
		try:
			response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
			assets = [x['name'] for x in response['assets']]  # release assets
			tag = response['tag_name']  # i.e. 'v1.0'
		except:  # fallback plan
			assets = ['dv2aug2.pt']
			tag = subprocess.check_output('git tag', shell=True).decode().split()[-1]

	name = file.name
	if name in assets:
		msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
		redundant = False  # second download option
		try:  # GitHub
			url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
			print(f'Downloading {url} to {file}...')
			torch.hub.download_url_to_file(url, file)
			assert file.exists() and file.stat().st_size > 1E6  # check
		except Exception as e:  # GCP
			print(f'Download error: {e}')
			assert redundant, 'No secondary mirror'
			url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
			print(f'Downloading {url} to {file}...')
			os.system(f'curl -L {url} -o {file}')  # torch.hub.download_url_to_file(url, weights)
		finally:
			if not file.exists() or file.stat().st_size < 1E6:  # check
				file.unlink(missing_ok=True)  # remove partial downloads
				print(f'ERROR: Download failure: {msg}')
			print('')
			return

def git_describe(path=Path(__file__).parent):  # path must be a directory
	# return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
	s = f'git -C {path} describe --tags --long --always'
	try:
		return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
	except subprocess.CalledProcessError as e:
		return ''  # not a git repository

def date_modified(path=__file__):
	# return human-readable file modification date, i.e. '2021-3-26'
	t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
	return f'{t.year}-{t.month}-{t.day}'


# Select Device utility, borrowed from yolov7
def select_device(device='', batch_size=None):
	# device = 'cpu' or '0' or '0,1,2,3'
	s = f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
	cpu = device.lower() == 'cpu'
	if cpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
	elif device:  # non-cpu device requested
		os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
		assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

	cuda = not cpu and torch.cuda.is_available()
	if cuda:
		n = torch.cuda.device_count()
		if n > 1 and batch_size:  # check that batch_size is compatible with device_count
			assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
			space = ' ' * len(s)
		for i, d in enumerate(device.split(',') if device else range(n)):
			p = torch.cuda.get_device_properties(i)
			s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
	else:
		s += 'CPU\n'

	logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
	return torch.device('cuda:0' if cuda else 'cpu')

