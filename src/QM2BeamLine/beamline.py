import re

import fabio
import numpy as np
from nexusformat.nexus import (NeXusError, NXcollection, NXdata, NXentry,
                               NXfield, NXgoniometer, NXlink, NXmonitor,
                               NXsample, NXsource, nxopen)
from nexusformat.nexus.tree import natural_sort
from nxrefine.nxbeamline import NXBeamLine
from nxrefine.nxutils import SpecParser

prefix_pattern = re.compile(r'^([^.]+)(?:(?<!\d)|(?=_))')
file_index_pattern = re.compile(r'^(.*?)([0-9]*)[.](.*)$')
directory_index_pattern = re.compile(r'^(.*?)([0-9]*)$')


class QM2BeamLine(NXBeamLine):

    name = 'QM2'
    source_name = 'Cornell High-Energy Synchrotron'
    create_macro_enabled = False
    import_data_enabled = True

    def __init__(self, reduce=None, directory=None):
        super().__init__(reduce=reduce, directory=directory)

    def import_data(self, config_file, overwrite=False):
        self.config_file = nxopen(config_file)
        scans = self.raw_directory / self.sample / self.label
        y_size, x_size = self.config_file['f1/instrument/detector/shape']
        for scan in [s for s in scans.iterdir() if s.is_dir()]:
            scan_name = self.sample+'_'+scan.name+'.nxs'
            scan_file = self.base_directory / scan_name
            if scan_file.exists() and not overwrite:
                continue
            scan_directories = [s.name for s in scan.glob(f'{self.sample}_*')
                                if s.is_dir()]
            if len(scan_directories) == 0:
                continue
            scan_directory = self.base_directory / scan.name
            scan_directory.mkdir(exist_ok=True)
            if overwrite:
                mode = 'w'
            else:
                mode = 'a'
            with nxopen(scan_file, mode) as root:
                if 'entry' not in root:
                    root['entry'] = self.config_file['entry']
                i = 0
                for s in scan_directories:
                    scan_number = self.get_index(s, directory=True)
                    if scan_number:
                        i += 1
                        entry_name = f"f{i}"
                        if entry_name in root and not overwrite:
                            continue
                        root[entry_name] = self.config_file['f1']
                        entry = root[entry_name]
                        entry['scan_number'] = scan_number
                        entry['data'] = NXdata()
                        linkpath = '/entry/data/data'
                        linkfile = scan_directory / f'f{i:d}.h5'
                        entry['data'].nxsignal = NXlink(linkpath, linkfile)
                        entry['data/x_pixel'] = np.arange(x_size, dtype=int)
                        entry['data/y_pixel'] = np.arange(y_size, dtype=int)
                        self.image_directory = scan / s
                        frame_number = len(self.get_files())
                        entry['data/frame_number'] = np.arange(frame_number,
                                                               dtype=int)
                        entry['data'].nxaxes = [entry['data/frame_number'],
                                                entry['data/y_pixel'],
                                                entry['data/x_pixel']]

    def load_data(self, overwrite=False):
        if self.reduce.raw_data_exists() and not overwrite:
            return True
        try:
            self.scan_number = self.entry['scan_number'].nxvalue
            scan_directory = f"{self.sample}_{self.scan_number:03d}"
            self.image_directory = (self.raw_directory /
                                    self.sample / self.label /
                                    self.scan / scan_directory)
            entry_file = self.entry.nxname+'.h5temp'
            self.raw_file = self.directory / entry_file
            self.write_data()
            self.raw_file.rename(self.raw_file.with_suffix('.h5'))
            return True
        except NeXusError:
            return False

    def get_prefix(self):
        prefixes = []
        for filename in self.image_directory.iterdir():
            match = prefix_pattern.match(filename.stem)
            if match and filename.suffix in ['.cbf', '.tif', '.tiff']:
                prefixes.append(match.group(1).strip('-').strip('_'))
        try:
            return max(prefixes, key=prefixes.count)
        except ValueError:
            raise NeXusError(f'No image files found in {self.image_directory}')

    def get_index(self, name, directory=False):
        try:
            if directory:
                return int(directory_index_pattern.match(str(name)).group(2))
            else:
                return int(file_index_pattern.match(str(name)).group(2))
        except Exception:
            return None

    def get_files(self):
        prefix = self.get_prefix()
        return sorted(
            [str(f) for f in self.image_directory.glob(prefix+'*')
             if f.suffix in ['.cbf', '.tif', '.tiff']],
            key=natural_sort)

    def read_image(self, filename):
        im = fabio.open(str(filename))
        return im.data

    def read_images(self, filenames, shape):
        good_files = [str(f) for f in filenames if f is not None]
        if good_files:
            v0 = self.read_image(good_files[0])
            if v0.shape != shape:
                raise NeXusError(
                    f'Image shape of {good_files[0]} not consistent')
            v = np.empty([len(filenames), v0.shape[0], v0.shape[1]],
                         dtype=np.float32)
        else:
            v = np.empty([len(filenames), shape[0], shape[1]],
                         dtype=np.float32)
        v.fill(np.nan)
        for i, filename in enumerate(filenames):
            if filename:
                v[i] = self.read_image(filename)
        return v

    def initialize_entry(self, filenames):
        z_size = len(filenames)
        v0 = self.read_image(filenames[0])
        x = NXfield(range(v0.shape[1]), dtype=np.uint16, name='x_pixel')
        y = NXfield(range(v0.shape[0]), dtype=np.uint16, name='y_pixel')
        z = NXfield(np.arange(z_size), dtype=np.uint16, name='frame_number',
                    maxshape=(5000,))
        v = NXfield(name='data', shape=(z_size, v0.shape[0], v0.shape[1]),
                    dtype=np.float32,
                    maxshape=(5000, v0.shape[0], v0.shape[1]))
        return NXentry(NXdata(v, (z, y, x)))

    def write_data(self):
        filenames = self.get_files()
        with nxopen(self.raw_file, 'w') as root:
            root['entry'] = self.initialize_entry(filenames)
            z_size = root['entry/data/data'].shape[0]
            image_shape = root['entry/data/data'].shape[1:3]
            chunk_size = root['entry/data/data'].chunks[0]
            k = 0
            for i in range(0, z_size, chunk_size):
                files = []
                for j in range(i, min(i+chunk_size, z_size)):
                    if j == self.get_index(filenames[k]):
                        print('Processing', filenames[k])
                        files.append(filenames[k])
                        k += 1
                    elif k < len(filenames):
                        files.append(None)
                    else:
                        break
                root['entry/data/data'][i:i+len(files), :, :] = (
                    self.read_images(files, image_shape))

    def get_logs(self):
        spec_file = self.raw_directory / self.sample
        if not spec_file.exists():
            self.reduce.log(f"'{spec_file}' does not exist")
            raise NeXusError('SPEC file not found')
        scan_number = self.entry['scan_number'].nxvalue
        logs = SpecParser(spec_file).read(scan_number).NXentry[0]
        logs.nxclass = NXcollection
        if 'logs' in self.entry:
            del self.entry['logs']
        return logs

    def get_source(self):
        source = NXsource()
        source['name'] = self.source_name
        source['type'] = self.source_type
        source['probe'] = 'x-ray'
        return source

    def get_monitor(self):
        if self.logs is None or self.monitor not in self.logs['data']:
            return None
        frame_number = self.entry['data/frame_number']
        frames = frame_number.size
        data = self.logs[f'data/{self.monitor}'][:frames]
        # Remove outliers at beginning and end of frames
        data[0:2] = data[2]
        data[-2:] = data[-3]
        monitor = NXmonitor(NXfield(data, name=self.monitor), frame_number)
        if 'data/frame_time' in self.entry:
            monitor['frame_time'] = self.entry['data/frame_time']
        return monitor

    def get_goniometer(self):
        if self.logs is None:
            return None
        g = NXgoniometer()
        if 'phi' in self.logs.get('data', {}):
            phi_arr = self.logs['data/phi']
            g['phi'] = NXfield(phi_arr[0])
            g['phi'].attrs['step'] = phi_arr[1] - phi_arr[0]
            g['phi'].attrs['end'] = phi_arr[-1]
        if 'chi' in self.logs.get('positioners', {}):
            g['chi'] = 90.0 - self.logs['positioners/chi']
        if 'th' in self.logs.get('positioners', {}):
            g['theta'] = self.logs['positioners/th']
        return g if g.entries else None

    def get_sample(self):
        sample = NXsample()
        sample['name'] = self.sample
        sample['label'] = self.label
        if self.logs is not None and 'sampleT' in self.logs.get('data', {}):
            sample['temperature'] = self.logs['data/sampleT'].average()
            sample['temperature'].attrs['units'] = 'K'
        return sample

    def get_start_time(self):
        if self.logs is None or 'date' not in self.logs:
            return None
        return self.logs['date']
