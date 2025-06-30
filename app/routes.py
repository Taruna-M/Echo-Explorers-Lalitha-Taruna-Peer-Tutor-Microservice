from fastapi import APIRouter, HTTPException, Request
from app.schemas import MatchRequest, MatchResponse
from app.matcher import match_peers
from app.appError import AppError
from app.offlineDocs import generate_offline_docs


router = APIRouter()

@router.post("/match-peer-tutors", response_model=MatchResponse)
def match_peer_tutors(request: MatchRequest):
    """
    Endpoint to match peer tutors based on user ID, topic, and urgency level.
        Args:
            request (MatchRequest): The request body containing user ID, topic, and urgency level.
        Returns:
            MatchResponse: The response containing matched peers and their details.
        Raises:
            HTTPException: If user ID or topic is missing, or if an unexpected error occurs.
    """
    try:
        if ( request.user_id in ["", None] or request.topic in ["", None] ):
            raise AppError("User ID and topic are required", 400)
        
        result = match_peers(
            user_id=request.user_id,
            topic=request.topic,
            urgency_level=request.urgency_level
        )
        return MatchResponse(**result)
    
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/health")
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}

@router.get("/version")
def version():
    """
    Returns the version of the application.
    """
    return {"model_version": "1.0.0"}

@router.get("/offline-docs")
async def offline_docs(request: Request):
    """
    Generates offline documentation for the API.
    """
    return generate_offline_docs(request.app)
    