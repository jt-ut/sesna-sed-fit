from astropy.io import fits

def explore_fits(fits_path):
    
    # Open and explore the FITS file
    with fits.open(fits_path) as hdul:
        # Print the overall structure
        print("FITS Structure:")
        print(hdul.info())
        print("\n" + "="*60 + "\n")
        
        # Look at each HDU in detail
        for i, hdu in enumerate(hdul):
            print(f"HDU {i}: {hdu.name}")
            print(f"  Type: {type(hdu).__name__}")
            
            # Show header info
            if hdu.header:
                print(f"  Key header cards:")
                for key in list(hdu.header.keys())[:15]:  # First 15 keys
                    print(f"    {key}: {hdu.header[key]}")
            
            # Show data info
            if hdu.data is not None:
                print(f"\n  Data shape: {hdu.data.shape}")
                print(f"  Data dtype: {hdu.data.dtype}")
                
                # If it's a table, show columns
                if hasattr(hdu.data, 'columns'):
                    print(f"  Column names: {hdu.data.columns.names}")
                    print(f"  First few rows:")
                    print(hdu.data[:3])  # First 3 rows
                else:
                    # Array data
                    print(f"  Data sample: {hdu.data.flat[:10]}")
            
            print("\n" + "-"*60 + "\n")

