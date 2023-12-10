## Building mnist3d on macOS

You may run into these errors when trying to build mnist3d:

```
# github.com/g3n/engine/audio/vorbis
/Users/foo/go/pkg/mod/github.com/g3n/engine@v0.2.0/audio/vorbis/vorbis.go:19:11: fatal error: 'codec.h' file not found
 #include "codec.h"
          ^~~~~~~~~
1 error generated.
# github.com/g3n/engine/audio/al
/Users/foo/go/pkg/mod/github.com/g3n/engine@v0.2.0/audio/al/al.go:20:11: fatal error: 'al.h' file not found
 #include "al.h"
          ^~~~~~
1 error generated.
```

You need to install `libvorbis` and `openal-soft`:

```
% brew install libvorbis openal-soft
```

Then export the paths to build it:

```
% export CGO_CFLAGS="-I$(brew --prefix)/include/vorbis -I$(brew --prefix)/include/ -I$(brew --prefix)/opt/openal-soft/include/AL" CGO_LDFLAGS="-L$(brew --prefix)/lib -L$(brew --prefix)/opt/openal-soft/lib"
% go build
# github.com/danaugrs/go-tsne/examples/mnist3d
ld: warning: directory not found for option '-L/usr/local/opt/openal-soft/lib'
```

The warning can be ignored (it's a hardcoded path in some makefile, but we are setting `-L` ourselves).
