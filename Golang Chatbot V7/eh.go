// eh Stands for Error Handler:
// Makes sure to print stuff via the Error Handler
package main

import (
	"fmt"
	"strconv"
)

// A function to locate what and where an error happenned
func ErrorHandler(quepaso string, dondepaso string){
	fmt.Printf("ERROR : %s : Funcion %s",quepaso,dondepaso)
}

// Function That Prints the length of a list
func PrintLength(length int) string {
	return strconv.Itoa(length)
}

// A function that prints all items in a list of strings
// If option == 1 , prints the list with 1 space in between
// If option != 1 , Prints the list with no spaces in between
func PrintList(list []string, option int){
	if option == 1 {
		PrintText(" Printing List with 1 Line in between",1)
		for _,val := range  list {
			fmt.Println(val)
		}
	}	else  {
		PrintText(" Printing List with no Space in between" ,1)
		for _,val := range  list {
			fmt.Print(val)
		}
	}
}

// A function that prints []float64
func PrintF64( list []float64, option int ) {
	if option == 1 {
		for _,val := range  list {
			fmt.Println(val)
		}
	}	else  {
		for _,val := range  list {
			fmt.Print(val)
			fmt.Print(",")
		}
	}
}

// A function that prints numbers 
// Receives a number and uses strconv to turn it into a print statement of said number
func PrintNumber(number int){
	fmt.Println(number)
}

// A function that prints strings 
// Receives a text and uses strconv to turn it into a print statement of said text
// If option == 1 , prints the list with 1 space in between
// If option != 1 , Prints the list with no spaces in between
func PrintText(text string, option int){
	if option == 1 {
		fmt.Println(text)
	} else  {
			fmt.Print(text)
	}
}

// A function that prints the keys of a Key: String map 
func PrintKeys( maps map[string]bool){
	for keys := range maps {
		fmt.Println(keys)
	}
}

// A function that prints the content of a map for a given Key (unique word) ,
// used for the One Hot Encoding Function
// Option 1: Calls for a single print of a given value
// Option != 1 : Calls for the print of all the values
func PrintMap_OHE( maps map[string][]bool , key string ) {
		fmt.Println(maps[key])
}