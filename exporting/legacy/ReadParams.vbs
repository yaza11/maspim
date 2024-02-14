' Function to read parameters from a file into a Dictionary object
Function ReadParametersFromFile(filePath)
    Dim fso, file, line, params, key, value
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set params = CreateObject("Scripting.Dictionary")

    ' Check if the file exists
    If fso.FileExists(filePath) Then
        Set file = fso.OpenTextFile(filePath, 1, False)

        ' Read the file line by line
        Do Until file.AtEndOfStream
            line = file.ReadLine
            ' Split each line into key and value
            If InStr(line, "=") > 0 Then
                key = Split(line, "=")(0)
                value = Split(line, "=")(1)
                ' Store them in the Dictionary object
                params(key) = value
            End If
        Loop

        file.Close
    Else
        WScript.Echo "Parameters file not found."
    End If

    ' Return the Dictionary object containing the parameters
    Set ReadParametersFromFile = params
End Function
