const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:10000';

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

// Help helper for images (if needed)
export const getImageUrl = (filename) => {
  return `${API_BASE_URL}/uploads/${filename}`;
};