import argparse
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--Niter", dest="Niter",
                      help="Number of MCMC iterations", type=int, default=1000)

parser.add_argument("--resume", dest="resume", help="True if ", type=bool)

args = parser.parse_args()

Niter = args.Niter
resume = args.resume
if resume == None:
    print("Resume argument not entered. Exiting.")
    sys.exit()
print("Niter is {}".format(Niter))
print("Continue run? {}".format(resume == True))
