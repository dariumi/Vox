// Go через cgo (C ABI): синтез в stdout — сырой float32 PCM.
//
//	cd examples/go && go build -o vox-go . && LD_LIBRARY_PATH=../../build ./vox-go model.tmod voice.tvoice "Привет!" > out.f32
package main

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../build -lvox
#include <stdlib.h>
#include "vox.h"
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

func check(st C.vox_status, what string) {
	if st < 0 {
		fmt.Fprintf(os.Stderr, "%s: %s\n", what, C.GoString(C.vox_last_error()))
		os.Exit(1)
	}
}

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "usage: %s model.tmod voice.tvoice text\n", os.Args[0])
		os.Exit(1)
	}
	model, voicePath, text := C.CString(os.Args[1]), C.CString(os.Args[2]), C.CString(os.Args[3])
	defer C.free(unsafe.Pointer(model))
	defer C.free(unsafe.Pointer(voicePath))
	defer C.free(unsafe.Pointer(text))

	var e *C.vox_engine
	var v *C.vox_voice
	var s *C.vox_session
	check(C.vox_engine_open(model, nil, &e), "open")
	defer C.vox_engine_close(e)
	check(C.vox_voice_load(e, voicePath, &v), "voice")
	defer C.vox_voice_free(v)

	var p C.vox_gen_params
	C.vox_gen_params_default(&p)
	p.seed = 42
	check(C.vox_session_create(e, v, &p, &s), "session")
	defer C.vox_session_free(s)
	check(C.vox_session_feed_text(s, text), "feed")
	check(C.vox_session_end_text(s), "end")

	buf := make([]float32, 4096)
	total := 0
	for {
		var n C.size_t
		st := C.vox_session_pull(s, (*C.float)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)), &n)
		if st != C.VOX_OK {
			check(st, "pull")
			break
		}
		os.Stdout.Write(unsafe.Slice((*byte)(unsafe.Pointer(&buf[0])), int(n)*4))
		total += int(n)
	}
	fmt.Fprintf(os.Stderr, "%d samples @ %d Hz\n", total, uint32(C.vox_engine_sample_rate(e)))
}
