package pkg

import (
	"fmt"
	"time"
)

func Timing(names string, start time.Time) {
	elapsed := time.Since(start)
	fmt.Printf("%s took %10f\n", names, elapsed.Seconds())
}
