' the modified script to export mass list to a text file in multiple write operations (every 1000 spectra)
' warning: could throw an error 'run out of string space' if the number of peaks is too large

' Include the ReadParams.vbs file, needs absolute path if run from dataanalysis
ExecuteGlobal CreateObject("Scripting.FileSystemObject").OpenTextFile("ReadParams.vbs", 1).ReadAll()

' Call the function to read parameters
Dim paramsFilePath, params
' needs absolute path if run from dataanalysis
paramsFilePath = "parameters.txt"
Set params = ReadParametersFromFile(paramsFilePath)

' Activate the DataAnalysis application if it is not already active
Set DA = GetObject("","BDal.DataAnalysis.Application")
DA.Activate(0)

mz_start = CInt(params("Q1mass")) - CInt(params("Width"))
mz_end = CInt(params("Q1mass")) + CInt(params("Width"))
const spec_start = 1

Set DA = GetObject("","BDal.DataAnalysis.Application")
DA.Activate(0)

Set fso = CreateObject("Scripting.FileSystemObject")

For Each currentAnalysis in DA.Analyses:
    savePath = params("ExportDataPath") & "\" & currentAnalysis.Name & ".txt"

    For SpecCounter = currentAnalysis.Spectra.Count to 1 Step -1  ' deletes all previously generated spectra
        currentAnalysis.Spectra.Delete SpecCounter
    Next

    spec_end = currentAnalysis.Chromatograms.Item(1).Size

    Dim fso, logfile, line, buffer, count
    Dim spec, peak, var
    Dim specIndex

    buffer = "" ' Use a buffer to accumulate lines before writing
    buff = params("customMethod") & vbCrLf & "Spot Number;Number of Peaks;M/Z;Intensity;S/N" & vbCrLf

    For specIndex = spec_start To spec_end
        currentAnalysis.Spectra.Add specIndex, daProfileOnly
        currentAnalysis.Spectra(currentAnalysis.Spectra.Count).MassListFind mz_start, mz_end

        Set spec = currentAnalysis.spectra(1)
        For Each var In spec.Variables
            If var.Name = "Spot Number" Then
                buffer = buffer & var.Value
                Exit For ' Exit once the spot number is found
            End If
        Next

        line = ""
        count = 0
        For Each peak In spec.MSPeakList
            ' Combine conditions to reduce redundancy
            If peak.m_over_z > mz_start And peak.m_over_z < mz_end Then
                count = count + 1
                line = line & ";" & peak.m_over_z & ";" & peak.Intensity & ";" & peak.SignaltoNoise
            End If
        Next

        buffer = buffer & ";" & count & line & vbCrLf ' Append newline for each spectrum

        currentAnalysis.Spectra.Delete currentAnalysis.Spectra.Count ' Remove spectrum from compound list

        If specIndex Mod 1000 = 0 Then ' Write the accumulated data to file every 100 spectra
            Set logfile = fso.OpenTextFile(savePath, 8, True)
            logfile.Write buffer
            logfile.Close
            buffer = "" ' Clear the buffer
        End If

    Next

    ' Write the remaining data to file
    Set logfile = fso.OpenTextFile(savePath, 8, True)
    logfile.Write buffer
    logfile.Close
Next