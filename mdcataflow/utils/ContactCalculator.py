"""
ContactCalculator - MD Trajectory Contact Analysis

Author: Maximilian Salomon
Version: 0.1.0
Created with assistance from Claude-4-Sonnet and Cursor AI.
"""

import numpy as np
import warnings
from .ArrayHandler import ArrayHandler

class ContactCalculator:
    """
    Utility class for computing contact maps from distance arrays.
    All methods are static and can be used without instantiation.
    """
    
    @staticmethod
    def _calculate_output_dimensions(distances_shape, squareform, k):
        """Calculate output dimensions and check if conversion is needed."""
        needs_conversion = (len(distances_shape) == 3 and not squareform) or (len(distances_shape) == 2 and squareform)
        
        if needs_conversion:
            if len(distances_shape) == 3 and not squareform:
                # Square to condensed
                n_residues = distances_shape[1]
                if k == 0:
                    n_contacts = n_residues * (n_residues + 1) // 2
                else:
                    n_contacts = n_residues * (n_residues - k) // 2 - sum(range(k))
                output_shape = (distances_shape[0], n_contacts)
            else:
                # Condensed to square
                n_contacts = distances_shape[1]
                n_residues = k + int((-1 + np.sqrt(1 + 8*n_contacts)) / 2)
                output_shape = (distances_shape[0], n_residues, n_residues)
        else:
            output_shape = distances_shape
            n_residues = n_contacts = None
        
        return output_shape, n_residues

    @staticmethod
    def _create_output_array(use_memmap, contacts_path, output_shape, dtype='bool'):
        """Create output array (memmap or regular)."""
        if use_memmap:
            if contacts_path is None:
                raise ValueError("contacts_path must be provided when use_memmap=True")
            return np.memmap(contacts_path, dtype=dtype, mode='w+', shape=output_shape)
        else:
            return np.zeros(output_shape, dtype=dtype)

    @staticmethod
    def compute_contacts(distances, cutoff=4.5, use_memmap=False, contacts_path=None, 
                        chunk_size=None, squareform=True, k=0):
        """
        Compute contact maps from distance arrays.
        
        Parameters:
        -----------
        distances : numpy.ndarray
            Distance array (NxMxM for square form or NxP for condensed form)
        cutoff : float, default=4.5
            Distance cutoff for contacts (in Angstrom)
        use_memmap : bool, default=False
            Whether to use memory mapping
        contacts_path : str, optional
            Path for memory-mapped contact array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
        squareform : bool, default=True
            If True, output NxMxM. If False, output NxP (upper triangular)
        k : int, default=0
            Diagonal offset: 0=include diagonal, 1=exclude diagonal, >1=exclude additional diagonals
            
        Returns:
        --------
        numpy.ndarray
            Boolean contact array
        """
        # Calculate dimensions and conversion requirements
        output_shape, n_residues = ContactCalculator._calculate_output_dimensions(
            distances.shape, squareform, k)
        
        # Create output array
        contacts = ContactCalculator._create_output_array(use_memmap, contacts_path, output_shape)
        
        # Process in chunks
        if chunk_size is None:
            chunk_size = distances.shape[0]
        
        for i in range(0, distances.shape[0], chunk_size):
            end_idx = min(i + chunk_size, distances.shape[0])
            chunk_contacts = distances[i:end_idx] <= cutoff
            
            # Convert format if needed
            if len(distances.shape) == 3 and not squareform:
                chunk_contacts = ArrayHandler.squareform_to_condensed(chunk_contacts, k=k, chunk_size=chunk_size)
            elif len(distances.shape) == 2 and squareform:
                chunk_contacts = ArrayHandler.condensed_to_squareform(chunk_contacts, n_residues, k=k, chunk_size=chunk_size)
            
            contacts[i:end_idx] = chunk_contacts
        
        return contacts
    
    @staticmethod
    def _compute_frequency_chunks(contacts, chunk_size):
        """Compute frequency using chunk processing for memmap."""
        total_frames = contacts.shape[0]
        result_shape = list(contacts.shape[1:])
        result = np.zeros(result_shape, dtype=np.float32)
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk = contacts[i:end_idx]
            result += np.sum(chunk, axis=0)
        
        return result / total_frames

    @staticmethod
    def compute_contact_frequency(contacts, axis=0, chunk_size=None):
        """Compute contact frequency across all frames."""
        if ArrayHandler.is_memmap(contacts) and chunk_size is not None:
            return ContactCalculator._compute_frequency_chunks(contacts, chunk_size)
        else:
            return np.mean(contacts, axis=axis, dtype=np.float32)
    
    @staticmethod
    def compute_contact_differences(contacts1, contacts2, chunk_size=None):
        """
        Compute differences in contact frequencies between two sets.
        
        Parameters:
        -----------
        contacts1 : numpy.ndarray
            First set of contacts
        contacts2 : numpy.ndarray
            Second set of contacts
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
            
        Returns:
        --------
        numpy.ndarray
            Difference in contact frequencies
        """
        freq1 = ContactCalculator.compute_contact_frequency(contacts1, chunk_size=chunk_size)
        freq2 = ContactCalculator.compute_contact_frequency(contacts2, chunk_size=chunk_size)
        return freq1 - freq2
    
    @staticmethod
    def compute_contacts_per_frame(contacts, chunk_size=None):
        """
        Compute total contacts per frame.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Boolean contact array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)

        Returns:
        --------
        numpy.ndarray
            Total contacts per frame
        """
        if ArrayHandler.is_memmap(contacts) and chunk_size is not None:
            total_frames = contacts.shape[0]
            contacts_per_frame = np.zeros(total_frames, dtype=np.int32)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = contacts[i:end_idx]
                if len(chunk.shape) == 3:
                    contacts_per_frame[i:end_idx] = np.sum(chunk, axis=(1, 2))
                else:
                    contacts_per_frame[i:end_idx] = np.sum(chunk, axis=1)
            
            return contacts_per_frame
        else:
            if len(contacts.shape) == 3:
                return np.sum(contacts, axis=(1, 2))
            else:
                return np.sum(contacts, axis=1)
    
    @staticmethod
    def compute_avg_contacts_per_residue(contacts, chunk_size=None):
        """
        Compute average contacts per residue. Only works with square format.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Boolean contact array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
            
        Returns:
        --------
        numpy.ndarray
            Average contacts per residue
        """
        if len(contacts.shape) != 3:
            raise ValueError("This method only works with square format (NxMxM)")
            
        if ArrayHandler.is_memmap(contacts) and chunk_size is not None:
            total_frames = contacts.shape[0]
            contacts_per_residue_sum = np.zeros(contacts.shape[1], dtype=np.int32)
            
            for i in range(0, total_frames, chunk_size):
                end_idx = min(i + chunk_size, total_frames)
                chunk = contacts[i:end_idx]
                contacts_per_residue_sum += np.sum(np.sum(chunk, axis=2), axis=0)
            
            return contacts_per_residue_sum / total_frames
        else:
            return np.mean(np.sum(contacts, axis=2), axis=0)
    
    @staticmethod
    def _compute_std_chunks(contacts, frequency, chunk_size):
        """Compute standard deviation using chunk processing for memmap."""
        total_frames = contacts.shape[0]
        sum_sq = np.zeros(contacts.shape[1:], dtype=np.float32)
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk = contacts[i:end_idx].astype(np.float32)
            if len(frequency.shape) == 2:
                chunk_mean = frequency[np.newaxis, :, :]
            else:
                chunk_mean = frequency[np.newaxis, :]
            sum_sq += np.sum((chunk - chunk_mean) ** 2, axis=0)
        
        return np.sqrt(sum_sq / total_frames)

    @staticmethod
    def compute_frequency_std(contacts, frequency, chunk_size=None):
        """Compute standard deviation of contact frequencies."""
        if ArrayHandler.is_memmap(contacts) and chunk_size is not None:
            return ContactCalculator._compute_std_chunks(contacts, frequency, chunk_size)
        else:
            return np.std(contacts.astype(np.float32), axis=0)

    @staticmethod
    def compute_contact_matrix_stats(contacts, chunk_size=None):
        """
        Compute comprehensive statistics for contact matrices efficiently.
        
        Parameters:
        -----------
        contacts : numpy.ndarray
            Boolean contact array
        chunk_size : int, default=None
            Chunk size for memmap processing (None for no chunking)
            
        Returns:
        --------
        dict
            Dictionary containing various statistics
        """
        stats = {}
        stats['frequency'] = ContactCalculator.compute_contact_frequency(contacts, chunk_size=chunk_size)
        stats['contacts_per_frame'] = ContactCalculator.compute_contacts_per_frame(contacts, chunk_size)
        if len(contacts.shape) == 3:
            stats['avg_contacts_per_residue'] = ContactCalculator.compute_avg_contacts_per_residue(contacts, chunk_size)
        stats['frequency_std'] = ContactCalculator.compute_frequency_std(contacts, stats['frequency'], chunk_size)
        return stats

    @staticmethod
    def _create_dynamic_mask(frequency, min_frequency, max_frequency):
        """Create mask for dynamic contacts."""
        mask = (frequency >= min_frequency) & (frequency <= max_frequency)
        n_selected = np.sum(mask.flatten())
        
        if n_selected == 0:
            warnings.warn("No contacts found within the specified frequency range. "
                         f"min_frequency={min_frequency}, max_frequency={max_frequency}")
        
        return mask, n_selected

    @staticmethod
    def _handle_feature_def(feature_def, mask):
        """Handle feature definition for dynamic contacts."""
        if feature_def is not None:
            if len(feature_def) != mask.size:
                warnings.warn(f"feature_def length ({len(feature_def)}) doesn't match "
                            f"mask size ({mask.size}). Returning mask instead.")
                return mask
            else:
                return np.array(feature_def)[mask.flatten()]
        return mask

    @staticmethod
    def _fill_dynamic_contacts_memmap(contacts, filtered_contacts, mask, chunk_size):
        """Fill filtered contacts using memmap processing."""
        if ArrayHandler.is_memmap(contacts):
            for i in range(0, contacts.shape[0], chunk_size):
                end_idx = min(i + chunk_size, contacts.shape[0])
                chunk = contacts[i:end_idx]
                filtered_contacts[i:end_idx] = chunk.reshape(chunk.shape[0], -1)[:, mask.flatten()]
        else:
            contacts_flat = contacts.reshape(contacts.shape[0], -1)
            for i in range(0, contacts.shape[0], chunk_size):
                end_idx = min(i + chunk_size, contacts.shape[0])
                filtered_contacts[i:end_idx] = contacts_flat[i:end_idx, mask.flatten()]

    @staticmethod
    def compute_dynamic_contacts(contacts, feature_def=None, min_frequency=0.1, max_frequency=0.9, 
                               use_memmap=False, reduced_contacts_path=None, chunk_size=1000):
        """Identify contacts with specific frequency range and return filtered contacts and indices/names."""
        frequency = ContactCalculator.compute_contact_frequency(contacts, chunk_size=chunk_size)
        mask, n_selected = ContactCalculator._create_dynamic_mask(frequency, min_frequency, max_frequency)
        contact_info = ContactCalculator._handle_feature_def(feature_def, mask)
        
        if use_memmap:
            if reduced_contacts_path is None:
                raise ValueError("reduced_contacts_path must be provided when use_memmap=True")
            
            filtered_contacts = np.memmap(reduced_contacts_path, dtype='bool', mode='w+',
                                        shape=(contacts.shape[0], n_selected))
            ContactCalculator._fill_dynamic_contacts_memmap(contacts, filtered_contacts, mask, chunk_size)
        else:
            contacts_flat = contacts.reshape(contacts.shape[0], -1)
            filtered_contacts = contacts_flat[:, mask.flatten()]
        
        return filtered_contacts, contact_info
    
    @staticmethod
    def compute_contact_variance(contacts, frequency=None, chunk_size=None):
        """Compute contact variance."""
        if frequency is None:
            frequency = ContactCalculator.compute_contact_frequency(contacts, chunk_size=chunk_size)
        return ContactCalculator.compute_frequency_std(contacts, frequency, chunk_size=chunk_size) ** 2
    
    @staticmethod
    def compute_contact_cv(contacts, frequency=None, chunk_size=None):
        """Compute coefficient of variation."""
        if frequency is None:
            frequency = ContactCalculator.compute_contact_frequency(contacts, chunk_size=chunk_size)
        variance = ContactCalculator.compute_contact_variance(contacts, frequency, chunk_size)
        return np.divide(np.sqrt(variance), frequency, out=np.zeros_like(variance), where=frequency>0)
    
    @staticmethod
    def compute_contact_transitions(contacts, chunk_size=None):
        """Compute number of state transitions."""
        if ArrayHandler.is_memmap(contacts) and chunk_size is not None:
            transitions = np.zeros(contacts.shape[1:], dtype=np.int32)
            for i in range(0, contacts.shape[0] - 1, chunk_size):
                end_idx = min(i + chunk_size, contacts.shape[0] - 1)
                chunk1 = contacts[i:end_idx]
                chunk2 = contacts[i+1:end_idx+1]
                transitions += np.sum(chunk1 != chunk2, axis=0)
        else:
            transitions = np.sum(np.diff(contacts, axis=0) != 0, axis=0)
        return transitions
    
    @staticmethod
    def compute_contact_stability(contacts, chunk_size=None):
        """Compute corrected stability based on transitions."""
        transitions = ContactCalculator.compute_contact_transitions(contacts, chunk_size)
        max_transitions = contacts.shape[0] - 1
        return 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0

    @staticmethod
    def compute_contact_variability(contacts, chunk_size=None):
        """Compute variability metrics for all contacts."""
        frequency = ContactCalculator.compute_contact_frequency(contacts, chunk_size=chunk_size)
        variance = ContactCalculator.compute_contact_variance(contacts, frequency, chunk_size)
        cv = ContactCalculator.compute_contact_cv(contacts, frequency, chunk_size)
        transitions = ContactCalculator.compute_contact_transitions(contacts, chunk_size)
        corrected_stability = ContactCalculator.compute_contact_stability(contacts, chunk_size)
        
        return {
            'frequency': frequency,
            'variance': variance,
            'cv': cv,
            'transitions': transitions,
            'corrected_stability': corrected_stability
        }
