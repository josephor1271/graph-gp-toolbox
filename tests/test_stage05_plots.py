"""The research plotting layer reads CSV only and reproduces its three PNGs."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from graphgp.benchmarking.report import save_csv


class Stage05Plots(unittest.TestCase):
    def test_reader_imports_no_training_or_data_stack(self):
        code="""import importlib.util,sys
spec=importlib.util.spec_from_file_location('plot05','experiments/05_plot_optimization.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert not any(x in sys.modules for x in ('torch','gpytorch','graphgp.benchmarking.stage05_worker','graphgp.benchmarking.stage05_cases'))
assert module.number({'x':'nan'},'x') is None
"""
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    def test_three_csv_only_figures_are_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);evidence=root/'csv';evidence.mkdir()
            row=dict(campaign='primary',case_id='s05_n100',arm='A_dense_direct',operation='mll',profile='balanced',rhs_width=0,
                     n=100,q=64,attempted=2,qualified_success=2,failed=0,timeouts=0,seconds_median=.1,seconds_min=.09,seconds_max=.11)
            save_csv(evidence/'summary.csv',[row])
            save_csv(evidence/'accuracy.csv',[dict(campaign='primary',case_id='s05_n100',arm='A_dense_direct',operation='mll',repeat=0,n=100,nll_abs_error=1e-12,heat_time_raw_abs_error=1e-12,wrong_sign_parameters=[])])
            save_csv(evidence/'entry_diagonal.csv',[dict(operation='entries',campaign='entries_batched',unique_columns=50,status='ok',operator={'operator_matmul_calls':24})])
            hashes=[]
            for directory in ('first','second'):
                output=root/directory
                subprocess.run([sys.executable,'experiments/05_plot_optimization.py','--evidence',str(evidence),'--output',str(output)],check=True,capture_output=True,text=True)
                images=sorted(output.glob('*.png'));self.assertEqual(len(images),3)
                hashes.append([hashlib.sha256(p.read_bytes()).hexdigest() for p in images])
            self.assertEqual(*hashes)


if __name__=='__main__':unittest.main()
