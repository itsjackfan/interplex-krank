# test_api.ps1

# -----------------------------
# Configuration Variables
# -----------------------------
$API_URL = "http://127.0.0.1:8000"
$USERNAME = "testuser"
$EMAIL = "testuser@example.com"
$PASSWORD = "password123"
$ARTICLE_TITLE = "Test Article"
$ARTICLE_CONTENT = "This is the content of the test article."

# -----------------------------
# Helper Functions
# -----------------------------

Function Extract-JsonField {
    param (
        [Parameter(Mandatory=$true)]
        [string]$JsonString,

        [Parameter(Mandatory=$true)]
        [string]$FieldName
    )

    try {
        $json = $JsonString | ConvertFrom-Json
        return $json.$FieldName
    }
    catch {
        Write-Host "[Error] Failed to extract field '$FieldName' from JSON." -ForegroundColor Red
        return $null
    }
}

Function Register-User {
    Write-Host "=============================================="
    Write-Host "1. Registering a New User"
    Write-Host "=============================================="

    $RegisterPayload = @{
        username = $USERNAME
        email    = $EMAIL
        password = $PASSWORD
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/register" -Method Post -ContentType "application/json" -Body $RegisterPayload -ErrorAction Stop
        Write-Host "[Success] User '$USERNAME' registered successfully." -ForegroundColor Green
    }
    catch {
        if ($_.Exception.Response.StatusCode.Value__ -eq 400) {
            # Corrected: Pass the stream as an argument to StreamReader
            $errorStream = $_.Exception.Response.GetResponseStream()
            $streamReader = New-Object System.IO.StreamReader($errorStream)
            $errorContent = $streamReader.ReadToEnd() | ConvertFrom-Json
            Write-Host "[Error] Username '$USERNAME' is already registered." -ForegroundColor Red
            Write-Host "Details: $($errorContent.detail | ConvertTo-Json -Depth 10)" -ForegroundColor Red
        }
        else {
            Write-Host "[Error] Failed to register user: $_" -ForegroundColor Red
        }
    }

    Write-Host ""
}


Function Login-User {
    Write-Host "=============================================="
    Write-Host "2. Logging In to Obtain Access Token"
    Write-Host "=============================================="

    $LoginBody = @{
        username = $USERNAME
        password = $PASSWORD
    }

    try {
        # Debug: Show the login payload
        Write-Host "Login Payload: $($LoginBody | ConvertTo-Json)" -ForegroundColor Cyan

        # Send the request and capture the response
        $response = Invoke-RestMethod -Uri "$API_URL/token" `
                                      -Method Post `
                                      -ContentType "application/x-www-form-urlencoded" `
                                      -Body $LoginBody `
                                      -ErrorAction Stop

        # Debug: Show the full response
        Write-Host "Full Login Response: $($response | ConvertTo-Json -Depth 10)" -ForegroundColor Cyan

        # Extract the access_token
        $AccessToken = $response.access_token
        
        if ([string]::IsNullOrEmpty($AccessToken)) {
            Write-Host "[Error] Access token is null or empty." -ForegroundColor Red
            return $null
        }

        Write-Host "[Success] Access Token Obtained: $AccessToken" -ForegroundColor Green
        return $AccessToken
    }
    catch {
        # Debug: Show the full error
        Write-Host "[Error] Failed to obtain access token: $_" -ForegroundColor Red
        return $null
    }

    Write-Host ""
}


Function Create-Article {
    param (
        $AccessToken
    )

    Write-Host "=============================================="
    Write-Host "3. Creating a New Article"
    Write-Host "=============================================="

    # Define the article payload
    $ArticlePayload = @{
        title     = $ARTICLE_TITLE
        content   = $ARTICLE_CONTENT
        timestamp = (Get-Date).ToString("o")  # ISO 8601 format
    } | ConvertTo-Json

    # Debug: Print the payload for inspection
    Write-Host "Article Payload: $($ArticlePayload)" -ForegroundColor Cyan

    # Prepare the Authorization header
    $AuthHeader = "Bearer $AccessToken"

    # Debug: Print the Authorization header to verify
    Write-Host "Authorization Header: $AuthHeader" -ForegroundColor Cyan

    try {
        # Perform the API request to create an article
        $response = Invoke-RestMethod -Uri "$API_URL/articles" `
                                      -Method Post `
                                      -Headers @{ Authorization = $AuthHeader } `
                                      -ContentType "application/json" `
                                      -Body $ArticlePayload `
                                      -ErrorAction Stop

        # Debug: Print the full response
        Write-Host "Full Article Response: $($response | ConvertTo-Json -Depth 10)" -ForegroundColor Cyan
        Write-Host "[Success] Article Created with ID: $($response._id)" -ForegroundColor Green
        return $response._id
    }
    catch {
        if ($_.Exception.Response.StatusCode.Value__ -eq 401) {
            Write-Host "[Error] Unauthorized: Access token may be invalid or expired." -ForegroundColor Red
        }
        elseif ($_.Exception.Response.StatusCode.Value__ -eq 400) {
            Write-Host "[Error] Bad Request: Please check the payload." -ForegroundColor Red
        }
        else {
            Write-Host "[Error] Failed to create article: $_" -ForegroundColor Red
        }
        return $null
    }

    Write-Host ""
}


Function Get-Article {
    param (
        [string]$AccessToken,
        [string]$ArticleId
    )

    Write-Host "=============================================="
    Write-Host "4. Retrieving the Created Article"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/articles/$ArticleId" -Method Get -Headers @{ Authorization = "Bearer $AccessToken" } -ErrorAction Stop
        Write-Host "[Success] Retrieved Article:"
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    catch {
        if ($_.Exception.Response.StatusCode.Value__ -eq 404) {
            Write-Host "[Error] Article with ID '$ArticleId' not found." -ForegroundColor Red
        }
        else {
            Write-Host "[Error] Failed to retrieve article: $_" -ForegroundColor Red
        }
    }

    Write-Host ""
}

Function Get-All-Articles {
    param (
        [string]$AccessToken
    )

    Write-Host "=============================================="
    Write-Host "5. Retrieving All Articles"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/articles/" -Method Get -Headers @{ Authorization = "Bearer $AccessToken" } -ErrorAction Stop
        if ($response.Count -eq 0) {
            Write-Host "[Info] No articles found for user '$USERNAME'." -ForegroundColor Yellow
        }
        else {
            Write-Host "[Success] Retrieved All Articles:"
            $response | ConvertTo-Json -Depth 10 | Write-Host
        }
    }
    catch {
        Write-Host "[Error] Failed to retrieve articles: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Update-Article {
    param (
        [string]$AccessToken,
        [string]$ArticleId
    )

    Write-Host "=============================================="
    Write-Host "6. Updating the Created Article"
    Write-Host "=============================================="

    $UpdatedTitle = "Updated Test Article"
    $UpdatedContent = "This is the updated content of the test article."

    $UpdatePayload = @{
        title     = $UpdatedTitle
        content   = $UpdatedContent
        timestamp = (Get-Date).ToString("o")  # ISO 8601 format
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/articles/$ArticleId" -Method Put -Headers @{ Authorization = "Bearer $AccessToken" } -ContentType "application/json" -Body $UpdatePayload -ErrorAction Stop
        Write-Host "[Success] Article Updated Successfully:" -ForegroundColor Green
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    catch {
        if ($_.Exception.Response.StatusCode.Value__ -eq 404) {
            Write-Host "[Error] Article not found or access denied." -ForegroundColor Red
        }
        else {
            Write-Host "[Error] Failed to update the article: $_" -ForegroundColor Red
        }
    }

    Write-Host ""
}

Function Query-Articles {
    param (
        [string]$AccessToken
    )

    Write-Host "=============================================="
    Write-Host "7. Querying Articles and Authors"
    Write-Host "=============================================="

    $QueryPayload = @{
        query = "test"
        limit = 5
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/query" -Method Post -Headers @{ Authorization = "Bearer $AccessToken" } -ContentType "application/json" -Body $QueryPayload -ErrorAction Stop
        Write-Host "[Success] Query Results:"
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    catch {
        Write-Host "[Error] Failed to perform query: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Health-Check {
    Write-Host "=============================================="
    Write-Host "8. Performing Health Check"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/health" -Method Get -ErrorAction Stop
        Write-Host "[Health Check] Response:"
        $response | ConvertTo-Json -Depth 10 | Write-Host
    }
    catch {
        Write-Host "[Error] Health check failed: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Test-MongoDB {
    Write-Host "=============================================="
    Write-Host "9. Testing MongoDB Connectivity"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/test/mongodb" -Method Get -ErrorAction Stop
        if ($response.mongodb -eq "Connection successful.") {
            Write-Host "[Success] MongoDB Connection: Connection successful." -ForegroundColor Green
        }
        else {
            Write-Host "[Error] MongoDB Connection Failed: $($response.mongodb)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "[Error] MongoDB Connection Failed: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Test-LLM {
    Write-Host "=============================================="
    Write-Host "10. Testing LLM Connectivity"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/test/llm" -Method Get -ErrorAction Stop
        if ($response.llm -eq "Connection successful.") {
            Write-Host "[Success] LLM Connection: Connection successful." -ForegroundColor Green
        }
        else {
            Write-Host "[Error] LLM Connection Failed: $($response.llm)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "[Error] LLM Connection Failed: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Test-Model {
    Write-Host "=============================================="
    Write-Host "11. Testing SentenceTransformer Model"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/test/model" -Method Get -ErrorAction Stop
        if ($response.model -eq "Loaded successfully.") {
            Write-Host "[Success] SentenceTransformer Model:" -ForegroundColor Green
            $response | ConvertTo-Json -Depth 10 | Write-Host
        }
        else {
            Write-Host "[Error] SentenceTransformer Model Test Failed: $($response.model)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "[Error] SentenceTransformer Model Test Failed: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Head-Request {
    Write-Host "=============================================="
    Write-Host "12. Sending HEAD Request to Root Endpoint"
    Write-Host "=============================================="

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/" -Method Head -ErrorAction Stop -Headers @{ "Accept" = "*/*" }
        Write-Host "[HEAD Request] Response Headers:"
        $response | Format-List
    }
    catch {
        Write-Host "[Error] HEAD Request Failed: $_" -ForegroundColor Red
    }

    Write-Host ""
}

Function Cleanup-TempFiles {
    # In this PowerShell script, we are not creating temporary files, so this function is optional.
    # If you decide to log responses to files, implement cleanup here.
    # Currently, all responses are handled in-memory.
    return
}

Function Run-AllTests {
    Register-User
    $AccessToken = Login-User

    if (-not $AccessToken) {
        Write-Host "[Terminating] Access Token not obtained. Exiting tests." -ForegroundColor Red
        return
    }

    $ArticleId = Create-Article($AccessToken)

    if (-not $ArticleId) {
        Write-Host "[Terminating] Article not created. Exiting tests." -ForegroundColor Red
        return
    }

    Get-Article -AccessToken $AccessToken -ArticleId $ArticleId
    Get-All-Articles -AccessToken $AccessToken
    Update-Article -AccessToken $AccessToken -ArticleId $ArticleId
    Query-Articles -AccessToken $AccessToken
    Health-Check
    Test-MongoDB
    Test-LLM
    Test-Model
    Head-Request

    Cleanup-TempFiles

    Write-Host "=============================================="
    Write-Host "API Testing Completed"
    Write-Host "=============================================="
}

# -----------------------------
# Execute All Tests
# -----------------------------
Run-AllTests

# -----------------------------
# Pause for User Review
# -----------------------------
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
