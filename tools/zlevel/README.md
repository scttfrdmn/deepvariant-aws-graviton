# zlevel: deflate level for make_examples outputs (#10)

`zlevel.c` is an LD_PRELOAD shim. Place it before zlib-ng (or any zlib) in `LD_PRELOAD`. When `DV_ZLEVEL` (0–9) is set, it replaces the level that callers ask for in `deflateInit2_`, `deflateInit_` and `deflateParams`. When unset it changes nothing.

make_examples writes its example, gVCF and small-model tfrecords gzip-compressed at the default level 6, through the zlib built into each extension module. zlib-ng interposes that zlib when preloaded, and so does this shim. Only the compressed bytes change; the records inside are the same.

    gcc -O2 -shared -fPIC -I<zlib include> -o libzlevel.so zlevel.c -ldl
    LD_PRELOAD=libjemalloc.so.2:libzlevel.so:/opt/zlib-ng/lib/libz.so.1 DV_ZLEVEL=1 make_examples ...

Where it applies: intermediate files only. Set it for make_examples, not for postprocess_variants, whose bgzipped VCF/gVCF are the deliverables.

Measurements: aws-realign#126 and this repository's #10.
