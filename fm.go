// fm Stands for :
// File Manager, Create and Write/Close Files, this module will create a text file with only the Data || Labels
// According to the needs. 

package main

import "os"

// Creates a file handle
func CreateFileHandle(name_extension string) *os.File {
	f, err := os.Create(name_extension)
	if err != nil {
		PrintText(err.Error(),1)
		return nil 
	}
	return f
}

// Writes information to the file handle
func WriteFile(data string, name_extension string){
	FileHandle := CreateFileHandle(name_extension)
	
	l, err := FileHandle.WriteString(data)
	if err != nil {
		PrintText(err.Error(),1)
		PrintNumber(l)
		FileHandle.Close()
		return
	}
	err = FileHandle.Close()
	if err != nil {
		PrintText(err.Error(),1)
		return
	}
}

