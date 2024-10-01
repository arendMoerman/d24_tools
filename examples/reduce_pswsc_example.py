import decode as dc
from d24_tools import tools as d24tools

if __name__ == "__main__":
    # This is an example script, which will probably FAIL when you try to run it, unless you have the following obsid zarr.zip file in your cwd.
    # Adjust the obsid and path to point to a zarr.zip relevant for your purposes.
    # For more detailed overviews of the tools, have a look at src/d24_tools/tools.py

    obsid = "20240715112833"

    bad_indices = [101, 103, 105] # These are random master indices to remove, just to show how the remove_bad_indices method works.

    path = os.path.join("./", f"dems_{_obs}.zarr.zip")

    da = dc.qlook.load_dems(path)

    da_sub = d24tools.remove_bad_indices(da, bad_indices)
    da_sub = d24tools.despike(da_sub)
    da_sub = d24tools.remove_overshoot(da_sub)
    
    avg,var,chan,freq = d24analysis.obs_to_nod_avg(da_sub, factor)

    num_nods = avg.shape[0]

    fig, ax = plt.subplots(num_nods,1)

    for i in range(num_nods):
        ax[i].errorbar(freq, avg[i,:], var[i,:], fmt="o")
    plt.show()
    
    fig, ax = plt.subplots(2,1)
    for i in range(num_nods):
        ax[0].plot(freq[arg_sort], avg[i,arg_sort], label=f"cycle {i}")
        #ax[1].plot(freq[arg_sort], avg[i,arg_sort]/np.sqrt(var[i,arg_sort]))
        ax[1].plot(freq[arg_sort], 1 / var[i, arg_sort])
    plt.legend()
    plt.show()

